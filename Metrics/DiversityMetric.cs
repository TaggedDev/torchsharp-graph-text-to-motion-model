using System;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace Text2Motion.TorchTrainer;

public class DiversityMetric
{
    // Guo et al. CVPR 2022 "Generating Diverse and Natural 3D Human Motions from Text"
    // https://github.com/EricGuo5513/text-to-motion/blob/main/utils/metrics.py (calculate_diversity)
    // Diversity = (1/S) · Σ ||f(m_i) - f(m_j)||₂  over S=300 independently-sampled random pairs
    public static float Compute(Tensor motionFeatures)
    {
        using var scope = NewDisposeScope();

        int n = (int)motionFeatures.shape[0];
        if (n < 2)
            return 0f;

        int S = Math.Min(300, n - 1);
        var device = motionFeatures.device;

        // Two independent without-replacement samples of size S (paper uses np.random.choice replace=False)
        var idx1 = randperm(n, dtype: ScalarType.Int64, device: device).narrow(0, 0, S);
        var idx2 = randperm(n, dtype: ScalarType.Int64, device: device).narrow(0, 0, S);

        var diff = motionFeatures.index_select(0, idx1) - motionFeatures.index_select(0, idx2);
        return diff.norm(dim: 1).mean().ToSingle();
    }
}
