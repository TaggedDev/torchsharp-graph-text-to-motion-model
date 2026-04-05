using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Training;

/// <summary>
/// DDPM noise scheduler with linear beta schedule.
/// Handles forward diffusion (q-sample), timestep sampling, and masked MSE loss.
/// </summary>
public sealed class DdpmScheduler
{
    private readonly int _numTimesteps;
    private Tensor _betas;
    private Tensor _alphas;
    private Tensor _alphasCumprod;
    private Tensor _sqrtAlphasCumprod;
    private Tensor _sqrtOneMinusAlphasCumprod;

    public DdpmScheduler(int numTimesteps = 1000, float betaStart = 0.0001f, float betaEnd = 0.02f)
    {
        _numTimesteps = numTimesteps;

        _betas = linspace(betaStart, betaEnd, numTimesteps, dtype: float64);
        _alphas = 1.0 - _betas;
        _alphasCumprod = _alphas.cumprod(dim: 0);
        _sqrtAlphasCumprod = _alphasCumprod.sqrt();
        _sqrtOneMinusAlphasCumprod = (1.0 - _alphasCumprod).sqrt();
    }

    public void To(Device device)
    {
        _betas = _betas.to(device);
        _alphas = _alphas.to(device);
        _alphasCumprod = _alphasCumprod.to(device);
        _sqrtAlphasCumprod = _sqrtAlphasCumprod.to(device);
        _sqrtOneMinusAlphasCumprod = _sqrtOneMinusAlphasCumprod.to(device);
    }

    /// <summary>
    /// Sample random timesteps in [0, T).
    /// </summary>
    public Tensor SampleTimesteps(int batchSize, Device device)
    {
        return randint(0, _numTimesteps, batchSize, dtype: int64, device: device);
    }

    /// <summary>
    /// Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    /// </summary>
    public Tensor QSample(Tensor x0, Tensor t, Tensor noise)
    {
        var sqrtA = _sqrtAlphasCumprod[t].to(float32).reshape(-1, 1, 1);
        var sqrtOmA = _sqrtOneMinusAlphasCumprod[t].to(float32).reshape(-1, 1, 1);
        return sqrtA * x0 + sqrtOmA * noise;
    }

    /// <summary>
    /// Masked MSE loss: mean over real frames only.
    /// predicted/target: [B, T, 263], mask: [B, T] (1.0 for real frames).
    /// </summary>
    public Tensor Loss(Tensor predicted, Tensor target, Tensor mask)
    {
        var diff = (predicted - target).pow(2);               // [B, T, 263]
        var masked = diff * mask.unsqueeze(-1);               // zero out padded frames
        var totalElements = mask.sum() * Data.Skeleton.FeatureDim;
        return masked.sum() / totalElements.clamp_min(1.0f);
    }

    /// <summary>
    /// Ancestral DDPM reverse sampling with optional classifier-free guidance.
    /// When <paramref name="guidanceScale"/> &gt; 1 the model is queried twice per
    /// step — once with <paramref name="cond"/>, once with <paramref name="nullCond"/>
    /// — and the predicted noise is extrapolated along the (cond − null) axis.
    /// Returns a [B, frames, featureDim] tensor in normalized feature space.
    /// </summary>
    public Tensor Sample(
        Module<Tensor, Tensor, Tensor, Tensor> model,
        Tensor cond,
        Tensor nullCond,
        int frames,
        int featureDim,
        Device device,
        float guidanceScale = 2.5f,
        float x0ClipValue = 0f,
        long? seed = null)
    {
        model.eval();

        using var noGrad = no_grad();
        using var outerScope = NewDisposeScope();

        if (seed.HasValue)
            manual_seed(seed.Value);

        int B = (int)cond.shape[0];
        bool useCfg = guidanceScale > 1f + 1e-6f;

        // For CFG we batch cond + nullCond together so the model makes a
        // single forward pass of size 2B per step.
        Tensor condBatched = useCfg ? cat(new[] { cond, nullCond }, dim: 0) : cond;

        var xt = randn(new long[] { B, frames, featureDim }, dtype: float32, device: device);

        for (int tInt = _numTimesteps - 1; tInt >= 0; tInt--)
        {
            using var stepScope = NewDisposeScope();

            Tensor predNoise;
            if (useCfg)
            {
                var xtBatched = cat(new[] { xt, xt }, dim: 0);
                var tBatched = full(new long[] { 2 * B }, tInt, dtype: int64, device: device);
                var bothNoise = model.forward(xtBatched, tBatched, condBatched);
                var chunks = bothNoise.chunk(2, dim: 0);
                var epsCond = chunks[0];
                var epsUncond = chunks[1];
                predNoise = epsUncond + guidanceScale * (epsCond - epsUncond);
            }
            else
            {
                var tTensor = full(new long[] { B }, tInt, dtype: int64, device: device);
                predNoise = model.forward(xt, tTensor, condBatched);
            }

            float alphaT = _alphas[tInt].to(float32).item<float>();
            float acpT = _alphasCumprod[tInt].to(float32).item<float>();
            float acpPrev = tInt > 0
                ? _alphasCumprod[tInt - 1].to(float32).item<float>()
                : 1f;
            float betaT = _betas[tInt].to(float32).item<float>();

            float sqrtAcpT = MathF.Sqrt(acpT);
            float sqrtOneMinusAcpT = MathF.Sqrt(1f - acpT);

            var x0 = (xt - sqrtOneMinusAcpT * predNoise) / sqrtAcpT;
            if (x0ClipValue > 0f)
                x0 = x0.clamp(-x0ClipValue, x0ClipValue);

            float coefX0 = MathF.Sqrt(acpPrev) * betaT / (1f - acpT);
            float coefXt = MathF.Sqrt(alphaT) * (1f - acpPrev) / (1f - acpT);
            var mean = coefX0 * x0 + coefXt * xt;

            Tensor next;
            if (tInt > 0)
            {
                float variance = betaT * (1f - acpPrev) / (1f - acpT);
                var noise = randn_like(xt);
                next = mean + MathF.Sqrt(variance) * noise;
            }
            else
            {
                next = mean;
            }

            // Promote the surviving tensor one scope up (into outerScope) so
            // stepScope can free all intermediates on dispose.
            xt = next.MoveToOuterDisposeScope();
        }

        return xt.detach().MoveToOuterDisposeScope();
    }
}
