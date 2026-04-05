using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Training;

/// <summary>
/// Single GCN layer: H' = SiLU(Â · H · W + b)
/// where Â is the symmetrically normalized adjacency with self-loops.
/// </summary>
public sealed class GraphConvLayer : Module<Tensor, Tensor>
{
    private readonly Linear _linear;
    private Tensor _adjNorm;

    public GraphConvLayer(string name, int inFeatures, int outFeatures, Tensor adjacency)
        : base(name)
    {
        _linear = Linear(inFeatures, outFeatures);
        _adjNorm = NormalizeAdjacency(adjacency);
        RegisterComponents();
    }

    /// <summary>
    /// Input: x [*, NumJoints, inFeatures]. Output: [*, NumJoints, outFeatures].
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        // Per-node linear transform
        var h = _linear.forward(x);             // [*, 22, outF]
        // Message passing via adjacency
        h = torch.matmul(_adjNorm, h);          // [22,22] @ [*, 22, outF]
        return functional.silu(h);
    }

    public void MoveBuffers(Device device)
    {
        _adjNorm = _adjNorm.to(device);
    }

    /// <summary>
    /// D^{-1/2} A D^{-1/2} symmetric normalization.
    /// </summary>
    private static Tensor NormalizeAdjacency(Tensor adj)
    {
        using var scope = NewDisposeScope();
        var degree = adj.sum(dim: 1);
        var dInvSqrt = degree.pow(-0.5);
        dInvSqrt = dInvSqrt.nan_to_num(0.0);
        var dMat = torch.diag(dInvSqrt);
        var result = dMat.mm(adj).mm(dMat);
        return result.MoveToOuterDisposeScope();
    }
}
