namespace Text2Motion.TorchTrainer;

public class TrainingConfig
{
    public int MaxEpochs { get; set; } = 100;
    public int PrintEveryEpoch { get; set; } = 5;
    public int RandomSeed { get; set; } = 42;
    public int BatchSize { get; set; } = 32;
    public int EvaluationBatchSize { get; set; } = 32;
    public float LearningRate { get; set; } = 1e-4f;
    public float WeightDecay { get; set; } = 0f;
    public string Device { get; set; } = "cuda";
}
