using System.Text.Json;

namespace Text2Motion.TorchTrainer;

public class TrainingMetricsService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private TrainerMetricsLog _log = new();
    private string _metricsPath = string.Empty;

    public TrainerMetricsLog Log => _log;

    public void Initialize(string metricsPath, bool loadCheckpoint)
    {
        _metricsPath = metricsPath;
        _log = LoadOrCreateLog(metricsPath, loadCheckpoint);
    }

    public void RecordEpoch(
        int epoch,
        float trainLoss,
        float trainMetric,
        float valLoss,
        float valMetric,
        float elapsedSeconds)
    {
        _log.Epochs.Add(epoch);
        _log.TrainLoss.Add(trainLoss);
        _log.TrainMetrics.Add(trainMetric);
        _log.ValidationLoss.Add(valLoss);
        _log.ValidationMetrics.Add(valMetric);
        _log.EpochSeconds.Add(elapsedSeconds);

        File.WriteAllText(_metricsPath, JsonSerializer.Serialize(_log, JsonOptions));
    }

    public TrainerMetricsLog CreateTestMetricsLog(int completedEpochs, float testLoss, float testMetric)
    {
        return new TrainerMetricsLog
        {
            Epochs = [completedEpochs],
            TestLoss = [testLoss],
            TestMetrics = [testMetric]
        };
    }

    public void RecordMotionMetrics(MotionEvalSnapshot snapshot)
    {
        _log.MotionEvalSnapshots.Add(snapshot);
        File.WriteAllText(_metricsPath, JsonSerializer.Serialize(_log, JsonOptions));
    }

    private static TrainerMetricsLog LoadOrCreateLog(string metricsPath, bool loadCheckpoint)
    {
        if (!loadCheckpoint || !File.Exists(metricsPath))
            return new TrainerMetricsLog();

        return JsonSerializer.Deserialize<TrainerMetricsLog>(
                   File.ReadAllText(metricsPath), JsonOptions)
               ?? new TrainerMetricsLog();
    }
}
