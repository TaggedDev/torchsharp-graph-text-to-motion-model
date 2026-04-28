namespace Text2Motion.TorchTrainer;

public class TrainingSettings
{
    public int MaxEpochs { get; set; } = 100;
    public int RandomSeed { get; set; } = 42;
    public int BatchSize { get; set; } = 32;
    public int EvaluationBatchSize { get; set; } = 32;
    public float LearningRate { get; set; } = 1e-4f;
    public float WeightDecay { get; set; } = 1e-4f;
    public string Device { get; set; } = "cuda";
    public string OutputRootPath { get; set; } = string.Empty;
    public bool LoadCheckpoint { get; set; }
    public int LoadRunNumber { get; set; }
}
