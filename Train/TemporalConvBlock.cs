using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Training;

/// <summary>
/// 1D temporal convolution block that operates along the time axis
/// for each joint independently. Used after spatial GCN blocks to
/// capture inter-frame temporal dependencies.
/// </summary>
public sealed class TemporalConvBlock : Module<Tensor, Tensor>
{
    private readonly int _numJoints;
    private readonly Conv1d _conv;
    private readonly Conv1d _pointwise;
    private readonly LayerNorm _norm;

    public TemporalConvBlock(string name, int channels, int numJoints, int kernelSize, int dilation)
        : base(name)
    {
        _numJoints = numJoints;
        int padding = dilation * (kernelSize - 1) / 2;

        _conv = Conv1d(channels, channels, kernelSize, dilation: dilation, padding: padding);
        _pointwise = Conv1d(channels, channels, 1);
        _norm = LayerNorm(channels);

        RegisterComponents();
    }

    /// <summary>
    /// Input: [B, T, numJoints, C]. Output: [B, T, numJoints, C].
    /// Applies temporal conv along T per joint with residual connection.
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        long B = x.shape[0];
        long T = x.shape[1];
        long C = x.shape[3];

        // [B, T, J, C] → [B, J, C, T] → [B*J, C, T]
        var h = x.permute(0, 2, 3, 1).reshape(B * _numJoints, C, T);

        h = functional.silu(_conv.forward(h));
        h = _pointwise.forward(h);

        // [B*J, C, T] → [B, J, C, T] → [B, T, J, C]
        h = h.reshape(B, _numJoints, C, T).permute(0, 3, 1, 2);

        // Residual + LayerNorm
        h = _norm.forward(h + x);
        return h;
    }
}
