using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Training;

/// <summary>
/// Spatio-temporal graph denoising network ε_θ(x_t, t, cond).
/// Each block alternates a spatial GCN over the 22-joint skeleton graph
/// with a temporal self-attention over the frame axis (per joint), so
/// information flows both across joints and across time.
/// </summary>
public sealed class GraphDenoiser : Module<Tensor, Tensor, Tensor, Tensor>
{
    private const int ClipDim = 512;
    private const int TimeEmbDim = 256;
    private const int MaxFrames = 512;

    private readonly int _numJoints;
    private readonly int _nodeHidden;
    private readonly int _flatHidden;
    private readonly int _numBlocks;

    private readonly Linear _inputProj;
    private readonly Linear _outputProj;
    private readonly Linear _timeMlp1;
    private readonly Linear _timeMlp2;
    private readonly Linear _condProj;

    private readonly ModuleList<GraphConvLayer> _gcnLayers;
    private readonly ModuleList<LayerNorm> _gcnNorms;
    private readonly ModuleList<TemporalAttentionBlock> _tempBlocks;

    // Learned null CLIP embedding, used as the unconditional token during
    // classifier-free guidance. Registered as a parameter so training can
    // shape it (with cond dropout) rather than using hard zeros.
    private readonly Parameter _nullCond;

    // Fixed sinusoidal frame positional embedding [MaxFrames, flatHidden].
    // Deterministic, so not part of the state dict — rebuilt on construction
    // and moved to device manually by MoveGcnBuffers (same pattern as
    // GraphConvLayer._adjNorm).
    private Tensor _framePosEmb;

    public GraphDenoiser(
        int numBlocks = 4,
        int nodeHidden = 64,
        int numHeads = 4)
        : base("GraphDenoiser")
    {
        _numJoints = Data.Skeleton.NumJoints;
        _nodeHidden = nodeHidden;
        _flatHidden = _numJoints * nodeHidden;
        _numBlocks = numBlocks;

        int featDim = Data.Skeleton.FeatureDim;

        _inputProj = Linear(featDim, _flatHidden);
        _outputProj = Linear(_flatHidden, featDim);

        _timeMlp1 = Linear(TimeEmbDim, _flatHidden);
        _timeMlp2 = Linear(_flatHidden, _flatHidden);
        _condProj = Linear(ClipDim, _flatHidden);

        var adj = Data.Skeleton.BuildAdjacency();
        _gcnLayers = new ModuleList<GraphConvLayer>();
        _gcnNorms = new ModuleList<LayerNorm>();
        _tempBlocks = new ModuleList<TemporalAttentionBlock>();
        for (int i = 0; i < numBlocks; i++)
        {
            _gcnLayers.Add(new GraphConvLayer($"gcn_{i}", nodeHidden, nodeHidden, adj));
            _gcnNorms.Add(LayerNorm(nodeHidden));
            _tempBlocks.Add(new TemporalAttentionBlock($"tattn_{i}", nodeHidden, numHeads));
        }

        _nullCond = new Parameter(zeros(1, ClipDim, dtype: float32));
        _framePosEmb = BuildFramePositionalEmbedding(MaxFrames, _flatHidden);

        RegisterComponents();
    }

    /// <summary>
    /// Unconditional CLIP token used as the null condition for CFG. Shape [1, 512].
    /// </summary>
    public Tensor NullCond => _nullCond;

    /// <summary>
    /// Predict noise: ε_θ(x_t, t, cond).
    /// x_t: [B, T, 263], t: [B] int64, cond: [B, 512].
    /// </summary>
    public override Tensor forward(Tensor xt, Tensor t, Tensor cond)
    {
        long B = xt.shape[0];
        long T = xt.shape[1];

        if (T > MaxFrames)
            throw new ArgumentException(
                $"Sequence length {T} exceeds MaxFrames={MaxFrames}");

        // Project input to per-node hidden features
        var h = _inputProj.forward(xt);                                // [B, T, flatHidden]

        // Timestep embedding
        var tEmb = SinusoidalEmbedding(t, TimeEmbDim);                 // [B, 256]
        tEmb = functional.silu(_timeMlp1.forward(tEmb));               // [B, flatHidden]
        tEmb = _timeMlp2.forward(tEmb);                                // [B, flatHidden]

        // Condition embedding
        var cEmb = _condProj.forward(cond);                            // [B, flatHidden]

        // Broadcast time + cond to every frame
        var combined = (tEmb + cEmb).unsqueeze(1);                     // [B, 1, flatHidden]
        h = h + combined;                                              // [B, T, flatHidden]

        // Add frame positional embedding (slice to current T)
        var posSlice = _framePosEmb.narrow(0, 0, T).unsqueeze(0);      // [1, T, flatHidden]
        h = h + posSlice;                                              // [B, T, flatHidden]

        // Reshape to per-joint view: [B, T, J, H]
        h = h.reshape(B, T, _numJoints, _nodeHidden);

        for (int i = 0; i < _numBlocks; i++)
        {
            // ---- Spatial substep: GCN over joints, per frame ----
            // [B, T, J, H] -> [B*T, J, H]
            var spatialIn = h.reshape(B * T, _numJoints, _nodeHidden);
            var gcnOut = _gcnLayers[i].forward(spatialIn);
            gcnOut = _gcnNorms[i].forward(gcnOut);
            var spatial = spatialIn + gcnOut;                          // residual
            h = spatial.reshape(B, T, _numJoints, _nodeHidden);

            // ---- Temporal substep: self-attention over frames, per joint ----
            // [B, T, J, H] -> [B, J, T, H] -> [B*J, T, H]
            var temporalIn = h.permute(0, 2, 1, 3).contiguous()
                              .reshape(B * _numJoints, T, _nodeHidden);
            var temporalOut = _tempBlocks[i].forward(temporalIn);      // residual is inside the block
            h = temporalOut.reshape(B, _numJoints, T, _nodeHidden)
                           .permute(0, 2, 1, 3).contiguous();          // back to [B, T, J, H]
        }

        // Project back to feature space
        h = h.reshape(B, T, _flatHidden);
        return _outputProj.forward(h);                                 // [B, T, 263]
    }

    public void MoveGcnBuffers(Device device)
    {
        foreach (var gcn in _gcnLayers)
            gcn.MoveBuffers(device);
        _framePosEmb = _framePosEmb.to(device);
    }

    /// <summary>
    /// Standard sinusoidal positional embedding (shared for diffusion
    /// timesteps and frame indices).
    /// </summary>
    private static Tensor SinusoidalEmbedding(Tensor positions, int dim)
    {
        int half = dim / 2;
        var freqs = exp(
            -Math.Log(10000.0) * arange(half, dtype: float32, device: positions.device) / half
        );
        var args = positions.to(float32).unsqueeze(1) * freqs.unsqueeze(0); // [N, half]
        return cat([args.cos(), args.sin()], dim: 1);                       // [N, dim]
    }

    private static Tensor BuildFramePositionalEmbedding(int maxFrames, int dim)
    {
        using var scope = NewDisposeScope();
        var positions = arange(maxFrames, dtype: float32);
        var emb = SinusoidalEmbedding(positions, dim);                      // [maxFrames, dim]
        return emb.MoveToOuterDisposeScope();
    }
}
