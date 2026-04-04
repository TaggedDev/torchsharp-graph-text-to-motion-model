using TorchSharp;
using static TorchSharp.torch;

namespace Data;

public static class Skeleton
{
    public const int NumJoints = 22;
    public const int FeatureDim = 263;

    // Kinematic tree: Parent[i] = parent joint of joint i. Root (pelvis) = -1.
    public static readonly int[] Parent =
    [
        -1, //  0: pelvis (root)
         0, //  1: right hip
         0, //  2: left hip
         0, //  3: spine1
         1, //  4: right knee
         2, //  5: left knee
         3, //  6: spine2
         4, //  7: right ankle
         5, //  8: left ankle
         6, //  9: spine3
         7, // 10: right foot
         8, // 11: left foot
         9, // 12: neck
         9, // 13: right collar
         9, // 14: left collar
        12, // 15: head
        13, // 16: right shoulder
        14, // 17: left shoulder
        16, // 18: right elbow
        17, // 19: left elbow
        18, // 20: right wrist
        19, // 21: left wrist
    ];

    // 21 undirected bones derived from the parent array
    public static readonly (int Src, int Dst)[] Edges =
        Parent
            .Select((parent, child) => (parent, child))
            .Where(e => e.parent >= 0)
            .ToArray();

    // 42 directed edges for GNN message passing (both directions)
    public static readonly (int Src, int Dst)[] BidirectionalEdges =
    [
        ..Edges,
        ..Edges.Select(e => (e.Dst, e.Src))
    ];

    // Joint groups for constraint losses
    public static readonly int[] LeftLeg = [0, 2, 5, 8, 11];
    public static readonly int[] RightLeg = [0, 1, 4, 7, 10];
    public static readonly int[] Spine = [0, 3, 6, 9, 12, 15];
    public static readonly int[] LeftArm = [9, 14, 17, 19, 21];
    public static readonly int[] RightArm = [9, 13, 16, 18, 20];

    // Bone length pairs for spatial distance constraint loss.
    // Each (A, B) is a bone whose length should stay constant across frames.
    public static readonly (int A, int B)[] BoneLengthPairs = Edges;

    // Angle triplets for angular constraint loss (CGDSPA-style).
    // Each (A, B, C) constrains the angle at joint B between bones B→A and B→C.
    public static readonly (int A, int B, int C)[] AngleTriplets =
    [
        ( 2,  0,  1), // pelvis: left hip – pelvis – right hip
        ( 0,  3,  6), // spine1 bend
        ( 3,  6,  9), // spine2 bend
        (13,  9, 14), // spine3: right collar – spine3 – left collar
        ( 0,  1,  4), // right hip angle
        ( 0,  2,  5), // left hip angle
        ( 1,  4,  7), // right knee
        ( 2,  5,  8), // left knee
        ( 4,  7, 10), // right ankle
        ( 5,  8, 11), // left ankle
        ( 9, 13, 16), // right collar–shoulder
        ( 9, 14, 17), // left collar–shoulder
        (13, 16, 18), // right shoulder–elbow
        (14, 17, 19), // left shoulder–elbow
        (16, 18, 20), // right elbow–wrist
        (17, 19, 21), // left elbow–wrist
    ];

    /// <summary>
    /// Builds a [22, 22] float32 adjacency matrix from the bidirectional edges + self-loops.
    /// Call once at startup and move to CUDA.
    /// </summary>
    public static Tensor BuildAdjacency()
    {
        var adj = torch.zeros(NumJoints, NumJoints, dtype: float32);
        foreach (var (src, dst) in BidirectionalEdges)
            adj[src, dst] = 1.0f;
        adj += torch.eye(NumJoints, dtype: float32);
        return adj;
    }
}
