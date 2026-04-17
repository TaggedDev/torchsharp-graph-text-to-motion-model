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
    public const int InputFeatures = 16;
    public const int HiddenFeatures = 32;
    public const int OutputFeatures = 16;

    private readonly Linear _inputProjection;
    private readonly Linear _outputProjection;

    public StubTextToMotionModel() : base(nameof(StubTextToMotionModel))
    {
        _inputProjection = Linear(InputFeatures, HiddenFeatures);
        _outputProjection = Linear(HiddenFeatures, OutputFeatures);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var hidden = _inputProjection.forward(input);
        hidden = functional.relu(hidden);
        return _outputProjection.forward(hidden);
    }
}
