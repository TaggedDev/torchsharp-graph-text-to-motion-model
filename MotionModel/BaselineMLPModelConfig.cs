namespace Text2Motion.TorchTrainer;

public class BaselineMLPModelConfig
{
    public int HiddenDim { get; set; } = 1024;
    public int NumHiddenLayers { get; set; } = 3;
}
