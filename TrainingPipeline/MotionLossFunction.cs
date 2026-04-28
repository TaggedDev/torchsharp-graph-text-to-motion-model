using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class MotionLossFunction
{
    private readonly float _velocityWeight;
    private readonly float _accelerationWeight;
    private readonly bool _useResidualMotion;

    public MotionLossFunction(
        float velocityWeight = 1.0f,
        float accelerationWeight = 0.1f,
        bool useResidualMotion = false)
    {
        _velocityWeight = velocityWeight;
        _accelerationWeight = accelerationWeight;
        _useResidualMotion = useResidualMotion;
    }

    public (Tensor totalLoss, Tensor posLoss, Tensor velLoss, Tensor accLoss) ComputeLoss(
        Tensor predicted,
        Tensor target,
        Tensor? firstFrameTarget = null)
    {
        Tensor posLoss;

        if (_useResidualMotion && firstFrameTarget is not null)
        {
            posLoss = ComputeResidualMotionLoss(predicted, target, firstFrameTarget);
        }
        else
        {
            posLoss = functional.mse_loss(predicted, target);
        }

        var velLoss = ComputeVelocityLoss(predicted, target);
        var accLoss = ComputeAccelerationLoss(predicted, target);

        var totalLoss = posLoss + _velocityWeight * velLoss + _accelerationWeight * accLoss;

        return (totalLoss, posLoss, velLoss, accLoss);
    }

    private static Tensor ComputeResidualMotionLoss(Tensor predicted, Tensor target, Tensor firstFrame)
    {
        // predicted: (B, T, 66) - predicted deltas
        // target: (B, T, 66) - actual deltas
        // firstFrame: (B, 66) - ground truth first frame
        return functional.mse_loss(predicted, target);
    }

    private static Tensor ComputeVelocityLoss(Tensor predicted, Tensor target)
    {
        long T = predicted.shape[1];
        if (T < 2)
            return tensor(0.0f);

        var predCurr = predicted.narrow(1, 0, T - 1);
        var predNext = predicted.narrow(1, 1, T - 1);
        var predVel = predNext - predCurr;

        var targetCurr = target.narrow(1, 0, T - 1);
        var targetNext = target.narrow(1, 1, T - 1);
        var targetVel = targetNext - targetCurr;

        return functional.mse_loss(predVel, targetVel);
    }

    private static Tensor ComputeAccelerationLoss(Tensor predicted, Tensor target)
    {
        long T = predicted.shape[1];
        if (T < 3)
            return tensor(0.0f);

        var predT0 = predicted.narrow(1, 0, T - 2);
        var predT1 = predicted.narrow(1, 1, T - 2);
        var predT2 = predicted.narrow(1, 2, T - 2);
        var predAcc = predT2 - 2 * predT1 + predT0;

        var targetT0 = target.narrow(1, 0, T - 2);
        var targetT1 = target.narrow(1, 1, T - 2);
        var targetT2 = target.narrow(1, 2, T - 2);
        var targetAcc = targetT2 - 2 * targetT1 + targetT0;

        return functional.mse_loss(predAcc, targetAcc);
    }
}
