namespace Text2Motion.TorchTrainer;

public class TrainerMetricsLog
{
    public List<int> Epochs { get; set; } = [];
    public List<float> TrainLoss { get; set; } = [];
    public List<float> ValidationLoss { get; set; } = [];
    public List<float> TestLoss { get; set; } = [];
    public List<float> TrainMetrics { get; set; } = [];
    public List<float> ValidationMetrics { get; set; } = [];
    public List<float> TestMetrics { get; set; } = [];
    public List<float> EpochSeconds { get; set; } = [];
    public List<MotionEvalSnapshot> MotionEvalSnapshots { get; set; } = [];
}