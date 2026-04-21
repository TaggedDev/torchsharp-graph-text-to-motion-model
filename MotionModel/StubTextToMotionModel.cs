using Microsoft.Extensions.Options;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

/// <summary>
/// Minimal TorchSharp module used to keep the trainer pipeline complete while
/// the real text-to-motion architecture is still under development.
/// </summary>
public sealed class StubTextToMotionModel : Module<Tensor, Tensor>
{
    private readonly Linear _inputProjection;
    private readonly Linear _outputProjection;

    public StubTextToMotionModel(IOptions<StubModelConfig> config) : base(nameof(StubTextToMotionModel))
    {
        var cfg = config.Value;
        _inputProjection = Linear(cfg.InputFeatures, cfg.HiddenFeatures);
        _outputProjection = Linear(cfg.HiddenFeatures, cfg.OutputFeatures);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var hidden = _inputProjection.forward(input);
        hidden = functional.relu(hidden);
        return _outputProjection.forward(hidden);
    }
}
