using TorchSharp;
using static TorchSharp.torch;

namespace Text2Motion.TorchTrainer;

public static class L2DistanceMetric
{
    public static float Compute(Tensor predicted, Tensor target)
    {
        using var scope = NewDisposeScope();
        var diff = predicted - target;
        var l2Norms = diff.norm(dim: 1);
        return l2Norms.mean().ToSingle();
    }
}
