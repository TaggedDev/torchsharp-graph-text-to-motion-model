using Microsoft.Extensions.Options;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public sealed class GcnSpatialTemporalModel : Module<Tensor, Tensor>
{
    private readonly Linear _inputProjection1;
    private readonly Linear _inputProjection2;
    private readonly LayerNorm _inputNorm;

    private readonly List<Linear> _gcnLinears = new();
    private readonly List<LayerNorm> _gcnNorms = new();
    private readonly List<Dropout> _gcnDropouts = new();

    private readonly List<Conv1d> _temporalConvs = new();
    private readonly List<BatchNorm1d> _temporalBns = new();
    private readonly List<Dropout> _temporalDropouts = new();

    private readonly Linear _outputProjection;

    private Tensor _adj;

    private readonly int _T, _J, _C, _Ct;

    public GcnSpatialTemporalModel(
        IOptions<GcnSpatialTemporalConfig> config,
        IOptions<DatasetSettings> dataset)
        : base(nameof(GcnSpatialTemporalModel))
    {
        var cfg = config.Value;
        var ds = dataset.Value;

        _T = ds.FixedFrames;      // 60
        _J = 22;                   // HumanML3D SMPL joints
        _C = cfg.JointFeatureDim;  // e.g. 64
        _Ct = _J * _C;             // e.g. 22 * 64 = 1408

        int outputDim = ds.FixedFrames * ds.FeatureDim; // 15780

        // Input projection (two-stage bottleneck)
        _inputProjection1 = Linear(ds.TextEmbeddingDim, 2048);  // 512 → 2048
        _inputProjection2 = Linear(2048, _T * _J * _C);         // 2048 → T*J*C
        _inputNorm = LayerNorm(_T * _J * _C);

        // Build adjacency matrix (not registered as parameter, manually moved to device)
        var adjFlat = BuildNormalizedAdjacency();
        _adj = from_array(adjFlat).reshape(_J, _J);

        // GCN layers
        for (int i = 0; i < cfg.NumGcnLayers; i++)
        {
            _gcnLinears.Add(Linear(_C, _C));
            _gcnNorms.Add(LayerNorm(_C));
            _gcnDropouts.Add(Dropout(cfg.DropoutRate));
        }

        // Temporal Conv1d layers
        for (int i = 0; i < cfg.NumTemporalLayers; i++)
        {
            _temporalConvs.Add(Conv1d(_Ct, _Ct, cfg.TemporalKernelSize, stride: 1, padding: cfg.TemporalPadding));
            _temporalBns.Add(BatchNorm1d(_Ct));
            _temporalDropouts.Add(Dropout(cfg.DropoutRate));
        }

        // Output projection
        _outputProjection = Linear(_T * _Ct, outputDim);

        // Manual registration BEFORE RegisterComponents()
        for (int i = 0; i < _temporalConvs.Count; i++)
        {
            register_module($"tconv_{i}", _temporalConvs[i]);
            register_module($"tbn_{i}", _temporalBns[i]);
            register_module($"tdrop_{i}", _temporalDropouts[i]);
        }

        for (int i = 0; i < _gcnDropouts.Count; i++)
        {
            register_module($"gcdrop_{i}", _gcnDropouts[i]);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // Step 1: Input projection (two-stage bottleneck)
        // input: (B, 512)
        var x = _inputProjection1.forward(input);  // (B, 2048)
        x = functional.gelu(x);

        x = _inputProjection2.forward(x);          // (B, T*J*C)
        x = _inputNorm.forward(x);
        x = functional.gelu(x);
        x = x.reshape(-1, _T, _J, _C);             // (B, T, J, C)

        // Step 2: Spatial GCN Block (shared weights across time, no inter-frame residual)
        var batchSize = x.shape[0];
        x = x.reshape(batchSize * _T, _J, _C);     // (B*T, J, C)

        var h = x;
        for (int i = 0; i < _gcnLinears.Count; i++)
        {
            // Batched GCN aggregation: (1, J, J) @ (B*T, J, C) → (B*T, J, C)
            var adj = _adj.to(x.device);
            var adjBatch = adj.unsqueeze(0);
            var agg = matmul(adjBatch, h);         // Graph aggregation
            var h_new = _gcnLinears[i].forward(agg);  // Linear(C, C)
            h_new = _gcnNorms[i].forward(h_new);
            h_new = functional.gelu(h_new);
            h_new = _gcnDropouts[i].forward(h_new);
            h = h + h_new;                         // Residual within GCN stack
        }

        x = h.reshape(batchSize, _T, _J, _C);      // (B, T, J, C)

        // Step 3: Prepare for Temporal Conv1d
        x = x.reshape(batchSize, _T, _J * _C);     // (B, T, Ct)
        x = x.permute(0, 2, 1);                    // (B, Ct, T) — Conv1d format

        // Step 4: Temporal Conv1d Block
        for (int i = 0; i < _temporalConvs.Count; i++)
        {
            var residual = x;
            x = _temporalConvs[i].forward(x);      // (B, Ct, T)
            x = _temporalBns[i].forward(x);
            x = functional.relu(x);
            x = _temporalDropouts[i].forward(x);
            x = x + residual;                      // Temporal residual
        }

        // Step 5: Output projection
        x = x.permute(0, 2, 1);                    // (B, T, Ct)
        x = x.reshape(batchSize, _T * _Ct);        // (B, T*Ct)
        x = _outputProjection.forward(x);          // (B, 15780)

        return x;
    }

    private static float[] BuildNormalizedAdjacency()
    {
        int J = 22;
        int[] parents = { -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19 };

        float[,] A = new float[J, J];

        // Self-loops
        for (int i = 0; i < J; i++)
            A[i, i] = 1f;

        // Tree edges (undirected)
        for (int j = 1; j < J; j++)
        {
            int p = parents[j];
            A[j, p] = 1f;
            A[p, j] = 1f;
        }

        // Degree
        float[] D = new float[J];
        for (int i = 0; i < J; i++)
            for (int k = 0; k < J; k++)
                D[i] += A[i, k];

        // D^(-1/2) A D^(-1/2)
        float[,] Anorm = new float[J, J];
        for (int i = 0; i < J; i++)
            for (int k = 0; k < J; k++)
                Anorm[i, k] = A[i, k] / (MathF.Sqrt(D[i]) * MathF.Sqrt(D[k]));

        // Flatten row-major
        float[] flat = new float[J * J];
        for (int i = 0; i < J; i++)
            for (int k = 0; k < J; k++)
                flat[i * J + k] = Anorm[i, k];

        return flat;
    }
}
