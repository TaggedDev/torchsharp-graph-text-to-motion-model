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
}
