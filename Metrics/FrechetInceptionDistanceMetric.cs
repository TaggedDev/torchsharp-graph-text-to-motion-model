using TorchSharp;
using static TorchSharp.torch;

namespace Text2Motion.TorchTrainer;

public class FrechetInceptionDistanceMetric
{
    // Heusel et al. 2017 "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
    // Applied to motion by Guo et al. CVPR 2022 "Generating Diverse and Natural 3D Human Motions from Text"
    // https://github.com/EricGuo5513/text-to-motion/blob/main/utils/metrics.py (calculate_frechet_distance)
    //
    // FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2·sqrtm(Σ_r·Σ_g))
    //
    // sqrtm(Σ_r·Σ_g) is numerically computed as:
    //   Tr(sqrtm(Σ_r·Σ_g)) = Σ_i sqrt(λ_i(M))  where M = Σ_r^½ · Σ_g · Σ_r^½
    // This uses the identity Tr(sqrtm(AB)) = Tr(sqrtm(A^½·B·A^½)) for SPD matrices A, B.
    // Σ_r^½ is computed via symmetric eigendecomposition: Σ = Q·Λ·Q^T → Σ^½ = Q·sqrt(Λ)·Q^T
    public static float Compute(Tensor muR, Tensor sigmaR, Tensor muG, Tensor sigmaG)
    {
        using var scope = NewDisposeScope();

        // ||μ_r − μ_g||²
        var muDiff = muR - muG;
        double meanDiffSq = muDiff.dot(muDiff).ToDouble();

        // Σ_r^½ via symmetric eigendecomposition: Σ_r = Q·Λ·Q^T → Σ_r^½ = Q·sqrt(Λ)·Q^T
        var (lambdaR, Q) = linalg.eigh(sigmaR);
        var sqrtSigmaR = (Q * lambdaR.relu().sqrt()).mm(Q.T);   // relu clamps FP-noise negatives

        // M = Σ_r^½ · Σ_g · Σ_r^½  (symmetric PSD by construction)
        var M = sqrtSigmaR.mm(sigmaG).mm(sqrtSigmaR);

        // Tr(sqrtm(Σ_r·Σ_g)) = Tr(sqrtm(M)) = Σ sqrt(max(0, λ_i(M)))
        var (lambdaM, _) = linalg.eigh(M);
        double trSqrtCovmean = lambdaM.relu().sqrt().sum().ToDouble();

        double fid = meanDiffSq
                   + trace(sigmaR).ToDouble()
                   + trace(sigmaG).ToDouble()
                   - 2.0 * trSqrtCovmean;
        return (float)fid;
    }
}
