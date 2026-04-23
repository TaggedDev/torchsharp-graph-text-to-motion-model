using static TorchSharp.torch;

namespace Text2Motion.TorchTrainer;

public class MultimodalDistanceMetric
{
    // Guo et al. CVPR 2022 "Generating Diverse and Natural 3D Human Motions from Text"
    // https://github.com/EricGuo5513/text-to-motion/blob/main/utils/metrics.py (calculate_matching_score)
    //
    // MMDist = (1/N) · Σ_i ||f_motion(m_i) − f_text(t_i)||₂
    // (mean of the diagonal of the cross-modal L2 distance matrix)
    //
    // Requires motion and text features in a SHARED 512-dim co-embedding space produced by
    // a pretrained contrastive encoder (MotionEncoderBiGRUCo + TextEncoderBiGRUCo).
    // Raw model outputs and CLIP text embeddings are in different spaces — L2 is undefined.
    public static float Compute(Tensor motionFeatures, Tensor textFeatures)
        => 0f;
}
