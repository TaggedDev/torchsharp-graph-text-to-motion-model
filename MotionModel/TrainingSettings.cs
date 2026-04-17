namespace Text2Motion.TorchTrainer;

public class TrainingSettings
{
    public string OutputRootPath { get; set; } = string.Empty;
    public bool LoadCheckpoint { get; set; }
    public int LoadRunNumber { get; set; }
}
