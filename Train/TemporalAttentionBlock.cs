using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Training;

/// <summary>
/// Pre-norm transformer block that applies multi-head self-attention along
/// the time axis, followed by a position-wise feed-forward network.
/// Input / output shape: [N, T, H] where N can be B*J (per-joint temporal
/// attention) and H == nodeHidden.
/// </summary>
public sealed class TemporalAttentionBlock : Module<Tensor, Tensor>
{
    private readonly LayerNorm _norm1;
    private readonly LayerNorm _norm2;
    private readonly MultiheadAttention _attn;
    private readonly Linear _ff1;
    private readonly Linear _ff2;

    public TemporalAttentionBlock(string name, int hiddenDim, int numHeads, int ffMult = 4)
        : base(name)
    {
        _norm1 = LayerNorm(hiddenDim);
        _norm2 = LayerNorm(hiddenDim);
        // TorchSharp 0.106.0 MultiheadAttention factory signature:
        //   (embedded_dim, num_heads, dropout=0, bias=true, add_bias_kv=false,
        //    add_zero_attn=false, kdim=null, vdim=null)
        // NOTE: no batch_first parameter in this version — attention expects
        // time-first [T, N, H] shape, so we transpose in forward().
        _attn = MultiheadAttention(hiddenDim, numHeads);

        int ffHidden = hiddenDim * ffMult;
        _ff1 = Linear(hiddenDim, ffHidden);
        _ff2 = Linear(ffHidden, hiddenDim);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass. x: [N, T, H]. Returns tensor of the same shape.
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        // x: [N, T, H]. TorchSharp 0.106.0 MultiheadAttention is time-first,
        // so transpose to [T, N, H] for the attention call and back afterward.
        var h = _norm1.forward(x);
        var hTimeFirst = h.transpose(0, 1).contiguous();
        // call(..) exposes the defaulted overload (forward requires all 6 args).
        var (attnOutTf, _) = _attn.call(hTimeFirst, hTimeFirst, hTimeFirst);
        var attnOut = attnOutTf.transpose(0, 1).contiguous();
        x = x + attnOut;

        // Pre-norm feed-forward with residual
        var h2 = _norm2.forward(x);
        h2 = _ff1.forward(h2);
        h2 = functional.silu(h2);
        h2 = _ff2.forward(h2);
        x = x + h2;

        return x;
    }
}
