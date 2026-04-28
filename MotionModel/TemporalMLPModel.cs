using Microsoft.Extensions.Options;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public sealed class TemporalMLPModel : Module<Tensor, Tensor>
{
    private readonly int _timeSteps;
    private readonly int _embeddingDim;
    private readonly int _outputDim;
    private readonly List<Linear> _linears = new();
    private readonly List<Dropout> _dropouts = new();
    private readonly Linear _outputLayer;
    private Tensor? _positionalEncoding;

    public TemporalMLPModel(IOptions<TemporalMLPModelConfig> config, IOptions<DatasetSettings> dataset)
        : base(nameof(TemporalMLPModel))
    {
        var cfg = config.Value;
        var ds = dataset.Value;

        _timeSteps = ds.FixedFrames;
        _embeddingDim = ds.TextEmbeddingDim;
        _outputDim = ds.JointFeatureDim;

        InitializePositionalEncoding();

        int hiddenDim = cfg.HiddenDim;
        int numLayers = cfg.NumHiddenLayers;
        float dropoutRate = cfg.DropoutRate;

        int inputDim = _embeddingDim;
        for (int i = 0; i < numLayers; i++)
        {
            _linears.Add(Linear(inputDim, hiddenDim));
            _dropouts.Add(Dropout(dropoutRate));
            inputDim = hiddenDim;
        }

        _outputLayer = Linear(hiddenDim, _outputDim);
        RegisterComponents();
    }

    private void InitializePositionalEncoding()
    {
        var encoding = zeros(_timeSteps, _embeddingDim);
        var position = arange(0, _timeSteps, dtype: float32).unsqueeze(1);
        var divTerm = exp(arange(0, _embeddingDim, 2, dtype: float32) * -(MathF.Log(10000.0f) / _embeddingDim));

        var sinVals = sin(position * divTerm);
        var cosVals = cos(position * divTerm);

        int evenDim = (_embeddingDim + 1) / 2;
        int oddDim = _embeddingDim / 2;

        for (int i = 0; i < evenDim && 2 * i < _embeddingDim; i++)
        {
            encoding.select(1, 2 * i).copy_(sinVals.select(1, i));
        }

        for (int i = 0; i < oddDim && 2 * i + 1 < _embeddingDim; i++)
        {
            encoding.select(1, 2 * i + 1).copy_(cosVals.select(1, i));
        }

        _positionalEncoding = encoding;
    }

    public override Tensor forward(Tensor textEmb)
    {
        var device = textEmb.device;
        long batchSize = textEmb.shape[0];

        var posEnc = _positionalEncoding!.to(device);
        var expandedEmb = textEmb.unsqueeze(1).expand(batchSize, _timeSteps, _embeddingDim);
        var withPos = expandedEmb + posEnc.unsqueeze(0);

        var output = new List<Tensor>();
        for (int t = 0; t < _timeSteps; t++)
        {
            var x = withPos.select(1, t);

            for (int i = 0; i < _linears.Count; i++)
            {
                x = _linears[i].forward(x);
                x = _dropouts[i].forward(x);
                x = functional.gelu(x);
            }

            x = _outputLayer.forward(x);
            output.Add(x);
        }

        var result = stack(output, dim: 1);
        return result;
    }
}
