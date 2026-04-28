namespace Text2Motion.TorchTrainer;

public class TemporalMLPModelConfig
{
    public int HiddenDim { get; set; } = 512;
    public int NumHiddenLayers { get; set; } = 3;
    public float DropoutRate { get; set; } = 0.2f;
}
