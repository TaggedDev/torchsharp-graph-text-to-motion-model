using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class MotionLossFunction
{
    private const float FirstFrameWeight = 0.1f;

    public MotionLossFunction()
    {
    }

    public (Tensor totalLoss, Tensor deltaLoss, Tensor firstFrameLoss) ComputeLoss(
        Tensor predicted,
        Tensor target)
    {
        var deltaLoss = ComputeDeltaLoss(predicted, target);
        var firstFrameLoss = ComputeFirstFrameLoss(predicted, target);

        var totalLoss = deltaLoss + FirstFrameWeight * firstFrameLoss;

        return (totalLoss, deltaLoss, firstFrameLoss);
    }

    private static Tensor ComputeDeltaLoss(Tensor predicted, Tensor target)
    {
        long T = predicted.shape[1];
        if (T < 2)
            return tensor(0.0f);

        var predDeltas = predicted.narrow(1, 1, T - 1) - predicted.narrow(1, 0, T - 1);
        var targetDeltas = target.narrow(1, 1, T - 1) - target.narrow(1, 0, T - 1);

        return functional.mse_loss(predDeltas, targetDeltas);
    }

    private static Tensor ComputeFirstFrameLoss(Tensor predicted, Tensor target)
    {
        var predFirstFrame = predicted.select(1, 0);
        var targetFirstFrame = target.select(1, 0);
        return functional.mse_loss(predFirstFrame, targetFirstFrame);
    }
}
