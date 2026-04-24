using TorchSharp;
using static TorchSharp.torch;

namespace Text2Motion.TorchTrainer;

public class WelfordCovarianceAccumulator : IDisposable
{
    private long _n = 0;
    private Tensor? _mean;
    private Tensor? _M2;

    public void Update(Tensor batch)
    {
        var b = batch.cpu().to(ScalarType.Float64);
        long m = b.shape[0];

        if (m == 0)
            return;

        var muB = b.mean(new long[] { 0 });
        var centered = b - muB;
        var M2b = centered.T.mm(centered);

        if (_n == 0)
        {
            _mean = muB.clone();
            _M2 = M2b.clone();
        }
        else
        {
            var delta = muB - _mean!;
            double w = (double)(_n * m) / (_n + m);
            var weighted = delta.outer(delta) * w;
            _M2 = _M2! + M2b + weighted;
            _mean = (_mean! * _n + muB * m) / (_n + m);
        }

        centered.Dispose();
        b.Dispose();
        muB.Dispose();
        M2b.Dispose();

        _n += m;
    }

    public (Tensor mean, Tensor covariance) Finalize()
    {
        if (_n < 2)
            throw new InvalidOperationException("Not enough samples to compute covariance");

        long D = _mean!.shape[0];
        var sigma = _M2! / (_n - 1);
        var epsI = eye(D, dtype: ScalarType.Float64) * 1e-6;
        sigma = sigma + epsI;

        return (_mean.clone(), sigma);
    }

    public void Dispose()
    {
        _mean?.Dispose();
        _M2?.Dispose();
    }
}
