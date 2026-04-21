using TorchSharp;
using static TorchSharp.torch;

namespace Text2Motion.TorchTrainer;

public class RPrecisionMetric
{
    public static float Compute(Tensor motionFeatures, Tensor textFeatures, int topK)
        => throw new NotImplementedException();
}
