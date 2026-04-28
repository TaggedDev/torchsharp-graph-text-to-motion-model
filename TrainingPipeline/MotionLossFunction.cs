using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class MotionLossFunction
{
    private readonly float _velocityWeight;
    private readonly float _accelerationWeight;

    public MotionLossFunction(float velocityWeight = 1.0f, float accelerationWeight = 0.1f)
    {
        _velocityWeight = velocityWeight;
        _accelerationWeight = accelerationWeight;
    }

    public (Tensor totalLoss, Tensor posLoss, Tensor velLoss, Tensor accLoss) ComputeLoss(
        Tensor predicted,
        Tensor target)
    {
        var posLoss = functional.mse_loss(predicted, target);

        var velLoss = ComputeVelocityLoss(predicted, target);
        var accLoss = ComputeAccelerationLoss(predicted, target);

        var smoothnessLoss = ComputeSmoothnessLoss(predicted);

        var totalLoss = posLoss + _velocityWeight * velLoss + _accelerationWeight * accLoss + smoothnessLoss;

        return (totalLoss, posLoss, velLoss, accLoss);
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

    private static Tensor ComputeSmoothnessLoss(Tensor predicted)
    {
        long T = predicted.shape[1];
        if (T < 2)
            return tensor(0.0f);

        var vel = predicted.narrow(1, 1, T - 1) - predicted.narrow(1, 0, T - 1);
        var velMagnitude = (vel * vel).sum(dim: 2);
        return velMagnitude.mean();
    }
}
