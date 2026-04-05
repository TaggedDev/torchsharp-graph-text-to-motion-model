using static TorchSharp.torch;

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
    /// Ancestral DDPM reverse sampling. Starts from pure noise and iteratively
    /// denoises conditioned on <paramref name="cond"/>.
    /// Returns a [B, frames, featureDim] tensor of generated motion in the
    /// network's normalized feature space (caller must denormalize).
    /// </summary>
    public Tensor Sample(
        GraphDenoiser model,
        Tensor cond,
        int frames,
        int featureDim,
        Device device,
        long? seed = null)
    {
        model.eval();

        using var noGrad = no_grad();
        using var outerScope = NewDisposeScope();

        if (seed.HasValue)
            manual_seed(seed.Value);

        int B = (int)cond.shape[0];
        var xt = randn(new long[] { B, frames, featureDim }, dtype: float32, device: device);

        for (int tInt = _numTimesteps - 1; tInt >= 0; tInt--)
        {
            using var stepScope = NewDisposeScope();

            var tTensor = full(new long[] { B }, tInt, dtype: int64, device: device);
            var predNoise = model.forward(xt, tTensor, cond);

            float alphaT = _alphas[tInt].to(float32).item<float>();
            float acpT = _alphasCumprod[tInt].to(float32).item<float>();
            float acpPrev = tInt > 0
                ? _alphasCumprod[tInt - 1].to(float32).item<float>()
                : 1f;
            float betaT = _betas[tInt].to(float32).item<float>();

            float sqrtAcpT = MathF.Sqrt(acpT);
            float sqrtOneMinusAcpT = MathF.Sqrt(1f - acpT);

            var x0 = (xt - sqrtOneMinusAcpT * predNoise) / sqrtAcpT;

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
