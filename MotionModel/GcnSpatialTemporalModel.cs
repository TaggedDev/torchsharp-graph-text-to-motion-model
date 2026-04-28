using Microsoft.Extensions.Options;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public sealed class GcnSpatialTemporalModel : Module<Tensor, Tensor>
{
    private readonly Linear _textToInitial;
    private readonly Linear _textTemporalEmbed;
    private readonly LayerNorm _inputNorm;

    private readonly List<Linear> _gcnLinears = new();
    private readonly List<LayerNorm> _gcnNorms = new();
    private readonly List<Dropout> _gcnDropouts = new();

    private readonly List<Conv1d> _temporalConvs = new();
    private readonly List<LayerNorm> _temporalNorms = new();
    private readonly List<Dropout> _temporalDropouts = new();

    private readonly Linear _rootHead;
    private readonly Linear _poseHead;

    private Tensor _adj;

    private readonly int _T, _J, _C, _Ct, _textDim;

    public GcnSpatialTemporalModel(
        IOptions<GcnSpatialTemporalConfig> config,
        IOptions<DatasetSettings> dataset)
        : base(nameof(GcnSpatialTemporalModel))
    {
        var cfg = config.Value;
        var ds = dataset.Value;

        _T = ds.FixedFrames;       // 60
        _J = 22;                   // HumanML3D SMPL joints
        _C = cfg.JointFeatureDim;  // e.g. 64
        _Ct = _J * _C;             // e.g. 1408
        _textDim = ds.TextEmbeddingDim; // 512

        // Text embedding → initial spatial features
        _textToInitial = Linear(_textDim, _T * _J * _C);
        _inputNorm = LayerNorm(_C);

        // Text temporal conditioning
        _textTemporalEmbed = Linear(_textDim, _C);

        // Build adjacency matrix
        var adjFlat = BuildNormalizedAdjacency();
        _adj = from_array(adjFlat).reshape(_J, _J);


        // GCN layers (per-frame spatial modeling)
        for (int i = 0; i < cfg.NumGcnLayers; i++)
        {
            _gcnLinears.Add(Linear(_C, _C));
            _gcnNorms.Add(LayerNorm(_C));
            _gcnDropouts.Add(Dropout(cfg.DropoutRate));
        }

        // Temporal layers with dilation
        int[] dilations = cfg.NumTemporalLayers switch
        {
            1 => new[] { 1 },
            2 => new[] { 1, 2 },
            3 => new[] { 1, 2, 4 },
            _ => new[] { 1, 2, 4, 8 }
        };

        for (int i = 0; i < dilations.Length; i++)
        {
            int dilation = dilations[i];
            int padding = (cfg.TemporalKernelSize - 1) * dilation / 2;
            _temporalConvs.Add(Conv1d(_Ct, _Ct, cfg.TemporalKernelSize, stride: 1, padding: padding, dilation: dilation));
            _temporalNorms.Add(LayerNorm(_Ct));
            _temporalDropouts.Add(Dropout(cfg.DropoutRate));
        }

        // Root (joint 0) and pose (joints 1-21) separate heads
        _rootHead = Linear(_C, 3);
        _poseHead = Linear(_C, 3);

        // Register components
        register_module("text_to_initial", _textToInitial);
        register_module("text_temporal", _textTemporalEmbed);
        register_module("input_norm", _inputNorm);

        for (int i = 0; i < _gcnLinears.Count; i++)
        {
            register_module($"gclinear_{i}", _gcnLinears[i]);
            register_module($"gcnorm_{i}", _gcnNorms[i]);
            register_module($"gcdrop_{i}", _gcnDropouts[i]);
        }

        for (int i = 0; i < _temporalConvs.Count; i++)
        {
            register_module($"tconv_{i}", _temporalConvs[i]);
            register_module($"tnorm_{i}", _temporalNorms[i]);
            register_module($"tdrop_{i}", _temporalDropouts[i]);
        }

        register_module("root_head", _rootHead);
        register_module("pose_head", _poseHead);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: (B, 512) text embedding
        var batchSize = input.shape[0];
        var device = input.device;

        // Step 1: Text → initial spatial features
        // (B, 512) → (B, T*J*C) → (B, T, J, C)
        var x = _textToInitial.forward(input);
        x = functional.gelu(x);
        x = x.reshape(batchSize, _T, _J, _C);

        // Step 2: Temporal conditioning - repeat text + positional encoding
        // (B, 512) → (B, T, 512) → (B, T, C)
        var textRep = input.unsqueeze(1).expand(batchSize, _T, -1);
        var textCond = _textTemporalEmbed.forward(textRep); // (B, T, C)

        // Add positional encoding to spatial features (sinusoidal, created fresh)
        var posEnc = CreatePositionalEncoding(_T, _C).to(device).unsqueeze(0).unsqueeze(2); // (1, T, 1, C)
        x = x + posEnc + textCond.unsqueeze(2); // (B, T, J, C) broadcast

        // Step 3: Spatial GCN (per frame)
        x = x.reshape(batchSize * _T, _J, _C); // (B*T, J, C)

        var h = x;
        for (int i = 0; i < _gcnLinears.Count; i++)
        {
            var adj = _adj.to(device).unsqueeze(0); // (1, J, J)
            var agg = matmul(adj, h); // Graph aggregation
            var h_new = _gcnLinears[i].forward(agg);
            h_new = _gcnNorms[i].forward(h_new);
            h_new = functional.gelu(h_new);
            h_new = _gcnDropouts[i].forward(h_new);
            h = h + h_new; // Residual
            adj.Dispose();
        }

        x = h.reshape(batchSize, _T, _J, _C); // (B, T, J, C)

        // Step 4: Temporal convolutions (dilated, for longer receptive field)
        x = x.reshape(batchSize, _T, _Ct); // (B, T, J*C)
        x = x.permute(0, 2, 1); // (B, J*C, T)

        for (int i = 0; i < _temporalConvs.Count; i++)
        {
            var residual = x;
            x = _temporalConvs[i].forward(x);
            x = x.permute(0, 2, 1); // → (B, T, J*C)
            x = _temporalNorms[i].forward(x);
            x = x.permute(0, 2, 1); // → (B, J*C, T)
            x = functional.relu(x);
            x = _temporalDropouts[i].forward(x);
            x = x + residual;
        }

        x = x.permute(0, 2, 1); // (B, T, J*C)
        x = x.reshape(batchSize, _T, _J, _C); // (B, T, J, C)

        // Step 5: Per-joint projection to 3D positions
        // Root head: (B, T, C) → (B, T, 3)
        var root = x.narrow(2, 0, 1).squeeze(2); // (B, T, C)
        var rootPos = _rootHead.forward(root); // (B, T, 3)

        // Pose head: (B, T, 21, C) → (B, T, 21, 3)
        var pose = x.narrow(2, 1, _J - 1); // (B, T, 21, C)
        pose = pose.reshape(batchSize * _T * (_J - 1), _C);
        var posePos = _poseHead.forward(pose); // (B*T*21, 3)
        posePos = posePos.reshape(batchSize, _T, _J - 1, 3);

        // Concatenate root + pose
        var output = cat(new[] { rootPos.unsqueeze(2), posePos }, dim: 2); // (B, T, 22, 3)
        output = output.reshape(batchSize, _T, _J * 3);

        return output;
    }

    private static Tensor CreatePositionalEncoding(int T, int C)
    {
        float[] pe = new float[T * C];

        for (int t = 0; t < T; t++)
        {
            for (int c = 0; c < C; c++)
            {
                float freq = MathF.Pow(10000, -2f * (c / 2) / C);
                if (c % 2 == 0)
                    pe[t * C + c] = MathF.Sin(t * freq);
                else
                    pe[t * C + c] = MathF.Cos(t * freq);
            }
        }

        return from_array(pe).reshape(T, C);
    }

    private static float[] BuildNormalizedAdjacency()
    {
        int J = 22;
        int[] parents = { -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19 };

        float[,] A = new float[J, J];

        for (int i = 0; i < J; i++)
            A[i, i] = 1f;

        for (int j = 1; j < J; j++)
        {
            int p = parents[j];
            A[j, p] = 1f;
            A[p, j] = 1f;
        }

        float[] D = new float[J];
        for (int i = 0; i < J; i++)
            for (int k = 0; k < J; k++)
                D[i] += A[i, k];

        float[,] Anorm = new float[J, J];
        for (int i = 0; i < J; i++)
            for (int k = 0; k < J; k++)
                Anorm[i, k] = A[i, k] / (MathF.Sqrt(D[i]) * MathF.Sqrt(D[k]));

        float[] flat = new float[J * J];
        for (int i = 0; i < J; i++)
            for (int k = 0; k < J; k++)
                flat[i * J + k] = Anorm[i, k];

        return flat;
    }
}
