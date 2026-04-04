using TorchSharp;
using static TorchSharp.torch;

namespace Data;

/// <summary>
/// Layout of the 263-dim vector:
///   [0:1]     root angular velocity (Y-axis) -> rootFeatures
///   [1:3]     root linear velocity (XZ) -> rootFeatures
///   [3:4]     root height (Y) -> rootFeatures
///   [4:67]    joint positions (21 joints × 3) -> jointPositions
///   [67:193]  joint rotations 6D cont. (21 joints × 6) -> jointRotations
///   [193:259] joint velocities (22 joints × 3) -> nodeVelocities
///   [259:263] foot contact (4 binary) -> footContact
/// </summary>
public static class FeatureSlicing
{
    public record struct GraphFeatures(
        Tensor RootFeatures,    // [B, T, 4]
        Tensor JointPositions,  // [B, T, 21, 3]
        Tensor JointRotations,  // [B, T, 21, 6]
        Tensor NodeVelocities,  // [B, T, 22, 3]
        Tensor FootContact      // [B, T, 4]
    );

    /// <summary>
    /// Slices the flat 263-dim tensor into per-joint feature tensors.
    /// Input: [B, T, 263]. All 263 values are accounted for exactly.
    /// </summary>
    public static GraphFeatures ToGraph(Tensor flat)
    {
        var B = flat.shape[0];
        var T = flat.shape[1];

        var rootFeatures   = flat[.., .., ..4];             
        var jointPositions = flat[.., .., 4..67].reshape(B, T, 21, 3);          
        var jointRotations = flat[.., .., 67..193].reshape(B, T, 21, 6);
        var nodeVelocities = flat[.., .., 193..259].reshape(B, T, 22, 3);         
        var footContact    = flat[.., .., 259..263];

        return new GraphFeatures(rootFeatures, jointPositions, jointRotations, nodeVelocities, footContact);
    }

    /// <summary>
    /// Inverse of ToGraph: reassembles the flat [B, T, 263] tensor.
    /// </summary>
    public static Tensor FromGraph(GraphFeatures g)
    {
        var B = g.RootFeatures.shape[0];
        var T = g.RootFeatures.shape[1];

        return torch.cat([
            g.RootFeatures,
            g.JointPositions.reshape(B, T, 63),
            g.JointRotations.reshape(B, T, 126),
            g.NodeVelocities.reshape(B, T, 66),
            g.FootContact
        ], dim: 2);
    }
}
