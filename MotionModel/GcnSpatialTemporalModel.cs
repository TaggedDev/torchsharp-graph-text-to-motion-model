using Microsoft.Extensions.Options;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public sealed class GcnSpatialTemporalModel : Module<Tensor, Tensor>
{
    public GcnSpatialTemporalModel(IOptions<GcnSpatialTemporalConfig> config, IOptions<DatasetSettings> dataset) : base(nameof(GcnSpatialTemporalConfig))
    {
        
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        throw new NotImplementedException();
    }
}