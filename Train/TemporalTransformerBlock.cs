using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Training;

/// <summary>
/// Self-attention temporal transformer block that operates along the time axis
/// for each joint independently. Replaces TemporalConvBlock to give the model
/// full-sequence receptive field instead of local dilated convolution.
/// Uses pre-norm (LayerNorm before attention/FFN) for training stability.
/// Implements attention via scaled_dot_product_attention for memory efficiency
/// (enables FlashAttention / memory-efficient attention kernels).
/// </summary>
public sealed class TemporalTransformerBlock : Module<Tensor, Tensor>
{
    private readonly int _numJoints;
    private readonly int _numHeads;
    private readonly int _headDim;

    private readonly LayerNorm _norm1;
    private readonly Linear _qkv;    // fused Q, K, V projection
    private readonly Linear _outProj;
    private readonly LayerNorm _norm2;
    private readonly Linear _ffn1;
    private readonly Linear _ffn2;

    public TemporalTransformerBlock(string name, int channels, int numJoints, int numHeads = 4)
        : base(name)
    {
        _numJoints = numJoints;
        _numHeads = numHeads;
        _headDim = channels / numHeads;

        _norm1 = LayerNorm(channels);
        _qkv = Linear(channels, channels * 3);   // fused QKV
        _outProj = Linear(channels, channels);
        _norm2 = LayerNorm(channels);
        _ffn1 = Linear(channels, channels * 4);
        _ffn2 = Linear(channels * 4, channels);

        RegisterComponents();
    }

    /// <summary>
    /// Input: [B, T, numJoints, C]. Output: [B, T, numJoints, C].
    /// Applies self-attention along T per joint with residual connections.
    /// </summary>
    public override Tensor forward(Tensor x)
    {
        long B = x.shape[0];
        long T = x.shape[1];
        long C = x.shape[3];
        long BJ = B * _numJoints;

        // [B, T, J, C] → [B*J, T, C]
        var h = x.permute(0, 2, 1, 3).reshape(BJ, T, C);

        // Self-attention with pre-norm + residual
        var normed = _norm1.forward(h);

        // Fused QKV: [BJ, T, 3*C] → split into Q, K, V each [BJ, T, C]
        var qkv = _qkv.forward(normed).chunk(3, dim: -1);
        // Reshape to [BJ, numHeads, T, headDim] for SDPA
        var q = qkv[0].reshape(BJ, T, _numHeads, _headDim).permute(0, 2, 1, 3);
        var k = qkv[1].reshape(BJ, T, _numHeads, _headDim).permute(0, 2, 1, 3);
        var v = qkv[2].reshape(BJ, T, _numHeads, _headDim).permute(0, 2, 1, 3);

        // Memory-efficient attention (uses FlashAttention when available)
        var attn = functional.scaled_dot_product_attention(q, k, v);

        // [BJ, numHeads, T, headDim] → [BJ, T, C]
        var attnOut = attn.permute(0, 2, 1, 3).reshape(BJ, T, C);
        attnOut = _outProj.forward(attnOut);
        h = h + attnOut;

        // FFN with pre-norm + residual
        var normed2 = _norm2.forward(h);
        var ffnOut = _ffn2.forward(functional.silu(_ffn1.forward(normed2)));
        h = h + ffnOut;

        // [B*J, T, C] → [B, J, T, C] → [B, T, J, C]
        return h.reshape(B, _numJoints, T, C).permute(0, 2, 1, 3);
    }
}
