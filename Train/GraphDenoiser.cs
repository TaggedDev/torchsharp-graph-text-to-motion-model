using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Training;

/// <summary>
/// Graph-based noise prediction network ε_θ(x_t, t, cond).
/// Projects 263-dim frames to per-node features, applies GCN blocks
/// over the skeleton graph, and projects back.
/// </summary>
public sealed class GraphDenoiser : Module<Tensor, Tensor, Tensor, Tensor>
{
    private const int ClipDim = 512;
    private const int TimeEmbDim = 256;

    private readonly int _numJoints;
    private readonly int _nodeHidden;
    private readonly int _flatHidden; // numJoints * nodeHidden

    private readonly Linear _inputProj;
    private readonly Linear _outputProj;
    private readonly Linear _timeMlp1;
    private readonly Linear _timeMlp2;
    private readonly Linear _condProj;
    private readonly ModuleList<GraphConvLayer> _gcnLayers;
    private readonly ModuleList<LayerNorm> _norms;
    private readonly ModuleList<TemporalConvBlock> _temporalLayers;

    public GraphDenoiser(
        int numGcnLayers = 4,
        int nodeHidden = 64,
        int temporalKernel = 3)
        : base("GraphDenoiser")
    {
        _numJoints = Data.Skeleton.NumJoints;
        _nodeHidden = nodeHidden;
        _flatHidden = _numJoints * nodeHidden;

        int featDim = Data.Skeleton.FeatureDim;

        _inputProj = Linear(featDim, _flatHidden);
        _outputProj = Linear(_flatHidden, featDim);

        // Timestep embedding MLP: TimeEmbDim → flatHidden
        _timeMlp1 = Linear(TimeEmbDim, _flatHidden);
        _timeMlp2 = Linear(_flatHidden, _flatHidden);

        // Condition projection: CLIP 512 → flatHidden
        _condProj = Linear(ClipDim, _flatHidden);

        // GCN layers + layer norms + temporal conv blocks
        var adj = Data.Skeleton.BuildAdjacency();
        _gcnLayers = new ModuleList<GraphConvLayer>();
        _norms = new ModuleList<LayerNorm>();
        _temporalLayers = new ModuleList<TemporalConvBlock>();
        for (int i = 0; i < numGcnLayers; i++)
        {
            _gcnLayers.Add(new GraphConvLayer($"gcn_{i}", nodeHidden, nodeHidden, adj));
            _norms.Add(LayerNorm(nodeHidden));
            int dilation = 1 << i; // 1, 2, 4, 8
            _temporalLayers.Add(new TemporalConvBlock($"temporal_{i}", nodeHidden, _numJoints, temporalKernel, dilation));
        }

        RegisterComponents();
    }

    /// <summary>
    /// Predict noise: ε_θ(x_t, t, cond).
    /// x_t: [B, T, 263], t: [B] int64, cond: [B, 512].
    /// Returns: [B, T, 263].
    /// </summary>
    public override Tensor forward(Tensor xt, Tensor t, Tensor cond)
    {
        long B = xt.shape[0];
        long T = xt.shape[1];

        // Project input to per-node hidden features
        var h = _inputProj.forward(xt);                                // [B, T, flatHidden]

        // Timestep embedding
        var tEmb = SinusoidalEmbedding(t, TimeEmbDim);                // [B, 256]
        tEmb = functional.silu(_timeMlp1.forward(tEmb));              // [B, flatHidden]
        tEmb = _timeMlp2.forward(tEmb);                               // [B, flatHidden]

        // Condition embedding
        var cEmb = _condProj.forward(cond);                            // [B, flatHidden]

        // Add time + condition (broadcast over T)
        var combined = (tEmb + cEmb).unsqueeze(1);                     // [B, 1, flatHidden]
        h = h + combined;                                              // [B, T, flatHidden]

        // Work in 4D: [B, T, numJoints, nodeHidden]
        h = h.reshape(B, T, _numJoints, _nodeHidden);

        // Alternating spatial GCN + temporal conv blocks
        for (int i = 0; i < _gcnLayers.Count; i++)
        {
            // --- Spatial: GCN over joints (per-frame) ---
            var hFlat = h.reshape(B * T, _numJoints, _nodeHidden);
            var residual = hFlat;
            hFlat = _gcnLayers[i].forward(hFlat);
            hFlat = _norms[i].forward(hFlat);
            hFlat = hFlat + residual;
            h = hFlat.reshape(B, T, _numJoints, _nodeHidden);

            // --- Temporal: conv along T (per-joint) ---
            h = _temporalLayers[i].forward(h);
        }

        // Project back to feature space
        h = h.reshape(B, T, _flatHidden);
        return _outputProj.forward(h);                                 // [B, T, 263]
    }

    public void MoveGcnBuffers(Device device)
    {
        foreach (var gcn in _gcnLayers)
            gcn.MoveBuffers(device);
    }

    /// <summary>
    /// Standard sinusoidal positional embedding for diffusion timesteps.
    /// </summary>
    private static Tensor SinusoidalEmbedding(Tensor t, int dim)
    {
        int half = dim / 2;
        var freqs = exp(
            -Math.Log(10000.0) * arange(half, dtype: float32, device: t.device) / half
        );
        var args = t.to(float32).unsqueeze(1) * freqs.unsqueeze(0);   // [B, half]
        return cat([args.cos(), args.sin()], dim: 1);            // [B, dim]
    }
}
