using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Training;

/// <summary>
/// Legacy (v1) noise-prediction network. Kept for loading pre-v2 checkpoints
/// under <c>checkpoints/*.pt</c>. This model has no temporal mixing and no
/// classifier-free guidance support — it is retained purely for A/B
/// comparison in the Visualize UI. All new training should use
/// <see cref="GraphDenoiser"/>.
///
/// Architecture: per-frame input projection → time+cond broadcast over T
/// → reshape [B*T, J, H] → 4 × (GCN + LayerNorm + residual) → reshape
/// [B, T, flatHidden] → output projection. Because time is collapsed into
/// the batch dimension during the GCN blocks, each frame is denoised
/// independently — this is the architectural flaw the v2 model fixes.
/// </summary>
public sealed class GraphDenoiserV1 : Module<Tensor, Tensor, Tensor, Tensor>
{
    private const int ClipDim = 512;
    private const int TimeEmbDim = 256;

    private readonly int _numJoints;
    private readonly int _nodeHidden;
    private readonly int _flatHidden;

    private readonly Linear _inputProj;
    private readonly Linear _outputProj;
    private readonly Linear _timeMlp1;
    private readonly Linear _timeMlp2;
    private readonly Linear _condProj;
    private readonly ModuleList<GraphConvLayer> _gcnLayers;
    private readonly ModuleList<LayerNorm> _norms;

    public GraphDenoiserV1(
        int numGcnLayers = 4,
        int nodeHidden = 64)
        : base("GraphDenoiser") // keep the legacy base name so v1 .pt files load
    {
        _numJoints = Data.Skeleton.NumJoints;
        _nodeHidden = nodeHidden;
        _flatHidden = _numJoints * nodeHidden;

        int featDim = Data.Skeleton.FeatureDim;

        _inputProj = Linear(featDim, _flatHidden);
        _outputProj = Linear(_flatHidden, featDim);

        _timeMlp1 = Linear(TimeEmbDim, _flatHidden);
        _timeMlp2 = Linear(_flatHidden, _flatHidden);
        _condProj = Linear(ClipDim, _flatHidden);

        var adj = Data.Skeleton.BuildAdjacency();
        _gcnLayers = new ModuleList<GraphConvLayer>();
        _norms = new ModuleList<LayerNorm>();
        for (int i = 0; i < numGcnLayers; i++)
        {
            _gcnLayers.Add(new GraphConvLayer($"gcn_{i}", nodeHidden, nodeHidden, adj));
            _norms.Add(LayerNorm(nodeHidden));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor xt, Tensor t, Tensor cond)
    {
        long B = xt.shape[0];
        long T = xt.shape[1];

        var h = _inputProj.forward(xt);
        var tEmb = SinusoidalEmbedding(t, TimeEmbDim);
        tEmb = functional.silu(_timeMlp1.forward(tEmb));
        tEmb = _timeMlp2.forward(tEmb);
        var cEmb = _condProj.forward(cond);

        var combined = (tEmb + cEmb).unsqueeze(1);
        h = h + combined;

        h = h.reshape(B * T, _numJoints, _nodeHidden);

        foreach (var (gcn, norm) in _gcnLayers.Zip(_norms))
        {
            var residual = h;
            h = gcn.forward(h);
            h = norm.forward(h);
            h = h + residual;
        }

        h = h.reshape(B, T, _flatHidden);
        return _outputProj.forward(h);
    }

    public void MoveGcnBuffers(Device device)
    {
        foreach (var gcn in _gcnLayers)
            gcn.MoveBuffers(device);
    }

    private static Tensor SinusoidalEmbedding(Tensor t, int dim)
    {
        int half = dim / 2;
        var freqs = exp(
            -Math.Log(10000.0) * arange(half, dtype: float32, device: t.device) / half
        );
        var args = t.to(float32).unsqueeze(1) * freqs.unsqueeze(0);
        return cat([args.cos(), args.sin()], dim: 1);
    }
}
