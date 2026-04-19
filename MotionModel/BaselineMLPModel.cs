using Microsoft.Extensions.Options;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public sealed class BaselineMLPModel : Module<Tensor, Tensor>
{
    private readonly List<Linear> _linears = new();
    private readonly List<LayerNorm> _layerNorms = new();
    private readonly Linear _outputLayer;

    public BaselineMLPModel(IOptions<ModelSettings> settings) : base(nameof(BaselineMLPModel))
    {
        var config = settings.Value;
        int outputDim = config.FixedFrames * config.FeatureDim;

        int inputDim = config.TextEmbeddingDim;
        int hiddenDim = config.HiddenDim;
        int numLayers = config.NumHiddenLayers;

        for (int i = 0; i < numLayers; i++)
        {
            _linears.Add(Linear(inputDim, hiddenDim));
            _layerNorms.Add(LayerNorm(hiddenDim));
            inputDim = hiddenDim;
        }

        _outputLayer = Linear(hiddenDim, outputDim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input;
        for (int i = 0; i < _linears.Count; i++)
        {
            x = _linears[i].forward(x);
            x = _layerNorms[i].forward(x);
            x = functional.gelu(x);
        }
        return _outputLayer.forward(x);
    }
}
