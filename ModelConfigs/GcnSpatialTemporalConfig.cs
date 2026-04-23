namespace Text2Motion.TorchTrainer;

public class GcnSpatialTemporalConfig
{
    public int JointFeatureDim { get; set; } = 64;
    public int NumGcnLayers { get; set; } = 3;
    public int NumTemporalLayers { get; set; } = 2;
    public int TemporalKernelSize { get; set; } = 3;
    public int TemporalPadding { get; set; } = 1;
    public float DropoutRate { get; set; } = 0.1f;
}