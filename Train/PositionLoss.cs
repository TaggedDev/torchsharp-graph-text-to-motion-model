using static TorchSharp.torch;

namespace Training;

/// <summary>
/// Position-based supervision adapted to this repository's motion layout.
/// It reconstructs 3D joint positions from the normalized 263-dim motion
/// representation. The primary training objective is Lpos: the average
/// L2 distance over all joints. Laux is computed as a diagnostic term
/// from the cited equations but is not used as the optimization target.
/// </summary>
public sealed class PositionLoss : IDisposable
{
    private readonly Device _device;
    private readonly Tensor _rootMean;
    private readonly Tensor _rootStd;
    private readonly Tensor _jointPosMean;
    private readonly Tensor _jointPosStd;
    private readonly Tensor _parentIndex;

    public PositionLoss(string processedRoot, Device device)
    {
        _device = device;

        var mean = ReadRawFloats(Path.Combine(processedRoot, "mean.bin"), Data.Skeleton.FeatureDim);
        var std = ReadRawFloats(Path.Combine(processedRoot, "std.bin"), Data.Skeleton.FeatureDim);

        _rootMean = tensor(new float[] { mean[3] }, dtype: float32, device: _device);
        _rootStd = tensor(new float[] { MathF.Max(std[3], 1e-5f) }, dtype: float32, device: _device);

        var jointPosMean = mean[4..67];
        var jointPosStd = std[4..67].Select(s => MathF.Max(s, 1e-5f)).ToArray();
        _jointPosMean = tensor(jointPosMean, dtype: float32, device: _device).reshape(1, 1, 21, 3);
        _jointPosStd = tensor(jointPosStd, dtype: float32, device: _device).reshape(1, 1, 21, 3);

        var parentIndex = Data.Skeleton.Parent[1..].Select(p => (long)p).ToArray();
        _parentIndex = tensor(parentIndex, dtype: int64, device: _device);
    }

     public (Tensor total, Tensor pos, Tensor aux, Tensor smooth, Tensor vel) Compute(
        Tensor predictedX0, Tensor targetX0, Tensor mask,
        float smoothWeight = 3.0f, float velocityWeight = 1.0f)
    {
        var predPos = ExtractJointPositions(predictedX0);
        var targetPos = ExtractJointPositions(targetX0);

        var validFrames = mask.unsqueeze(-1);
        var validJointCount = mask.sum().clamp_min(1.0f) * Data.Skeleton.NumJoints;
        var validBoneCount = mask.sum().clamp_min(1.0f) * (Data.Skeleton.NumJoints - 1);

        var l2PerJoint = (predPos - targetPos).pow(2).sum(dim: -1).add(1e-8f).sqrt();
        var posLoss = (l2PerJoint * validFrames).sum() / validJointCount;

        var predChildren = predPos[.., .., 1.., ..];
        var predParents = predPos.index_select(2, _parentIndex);
        var l1PerBone = (predChildren - predParents).abs().sum(dim: -1);
        var auxLoss = (l1PerBone * validFrames).sum() / validBoneCount;

        // Velocity loss (1st-order finite difference)
        var predVel = predPos[.., 1.., .., ..] - predPos[.., ..^1, .., ..];       // [B, T-1, 22, 3]
        var targetVel = targetPos[.., 1.., .., ..] - targetPos[.., ..^1, .., ..]; // [B, T-1, 22, 3]
        var velMask = mask[.., ..^1] * mask[.., 1..];                              // [B, T-1]
        var velCount = velMask.sum().clamp_min(1.0f) * Data.Skeleton.NumJoints;
        var velLoss = ((predVel - targetVel).pow(2).sum(dim: -1) * velMask.unsqueeze(-1)).sum() / velCount;

        // Acceleration smoothness loss (2nd-order finite difference)
        var accel = predVel[.., 1.., .., ..] - predVel[.., ..^1, .., ..];         // [B, T-2, 22, 3]
        var accelMask = mask[.., ..^2] * mask[.., 1..^1] * mask[.., 2..];         // [B, T-2]
        var accelCount = accelMask.sum().clamp_min(1.0f) * Data.Skeleton.NumJoints;
        var smoothLoss = (accel.pow(2).sum(dim: -1) * accelMask.unsqueeze(-1)).sum() / accelCount;

        var total = posLoss + auxLoss + smoothWeight * smoothLoss + velocityWeight * velLoss;
        return (total, posLoss, auxLoss, smoothLoss, velLoss);
    }

    private Tensor ExtractJointPositions(Tensor flat)
    {
        var rootHeight = flat[.., .., 3].unsqueeze(-1);
        rootHeight = rootHeight * _rootStd + _rootMean;

        var zeros = zeros_like(rootHeight);
        var root = cat([zeros, rootHeight, zeros], dim: -1).unsqueeze(2);

        var jointPos = flat[.., .., 4..67].reshape(flat.shape[0], flat.shape[1], 21, 3);
        jointPos = jointPos * _jointPosStd + _jointPosMean;

        return cat([root, jointPos], dim: 2);
    }

    private static float[] ReadRawFloats(string path, int expectedCount)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[expectedCount];
        Buffer.BlockCopy(bytes, 0, floats, 0, expectedCount * sizeof(float));
        return floats;
    }

    public void Dispose()
    {
        _rootMean.Dispose();
        _rootStd.Dispose();
        _jointPosMean.Dispose();
        _jointPosStd.Dispose();
        _parentIndex.Dispose();
    }
}
