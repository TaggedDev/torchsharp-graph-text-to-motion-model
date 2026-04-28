namespace Text2Motion.TorchTrainer;

public class DatasetSettings
{
    public string TrainSplitPath { get; set; } = string.Empty;
    public string ValidationSplitPath { get; set; } = string.Empty;
    public string TestSplitPath { get; set; } = string.Empty;
    public string JointsPath { get; set; } = string.Empty;
    public string EmbeddingsPath { get; set; } = string.Empty;
    public string NormalizationMeanPath { get; set; } = string.Empty;
    public string NormalizationStdPath { get; set; } = string.Empty;

    public int FixedFrames { get; set; } = 60;
    public int FullFeatureDim { get; set; } = 263;
    public int JointStartIdx { get; set; } = 4;
    public int JointEndIdx { get; set; } = 70;
    public int NumJoints { get; set; } = 22;
    public int JointFeatureDim { get; set; } = 66;
    public int FeatureDim { get; set; } = 66;
    public int TextEmbeddingDim { get; set; } = 512;
}
