using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Options;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class TextToMotionModelTrainer(IOptions<ModelSettings> options)
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly ModelSettings _settings = options.Value;

    public Task TrainAsync(CancellationToken token)
    {
        int maxEpochs = Math.Max(1, _settings.MaxEpochs);
        int printEveryEpoch = Math.Max(1, _settings.PrintEveryEpoch);

        SetRandomSeed(_settings.RandomSeed);

        string runDirectoryPath = InitializeRunDirectory(_settings);
        string checkpointsPath = Path.Combine(runDirectoryPath, "checkpoints");
        string metricsPath = Path.Combine(runDirectoryPath, "metrics.json");
        string settingsPath = Path.Combine(runDirectoryPath, "model-settings.json");

        Directory.CreateDirectory(checkpointsPath);
        File.WriteAllText(settingsPath, JsonSerializer.Serialize(_settings, JsonOptions));

        var metricsLog = new TrainerMetricsLog();
        var trainingState = new TrainingState();
        var model = CreateModel();
        var optimizer = torch.optim.Adam(model.parameters(), lr: _settings.LearningRate, weight_decay: _settings.WeightDecay);

        for (int epoch = 1; epoch <= maxEpochs; epoch++)
        {
            token.ThrowIfCancellationRequested();

            var epochTimer = Stopwatch.StartNew();
            Console.WriteLine($"Epoch {epoch}/{maxEpochs} started.");

            model.train();
            var trainingMetrics = RunStubEpoch(model, optimizer, batchSize: Math.Max(1, _settings.BatchSize), training: true);

            model.eval();
            var validationMetrics = RunStubEpoch(model, optimizer: null, batchSize: Math.Max(1, _settings.EvaluationBatchSize), training: false);

            epochTimer.Stop();

            trainingState.CompletedEpochs = epoch;
            trainingState.LatestLearningRate = _settings.LearningRate;

            metricsLog.Epochs.Add(epoch);
            metricsLog.TrainLoss.Add(trainingMetrics.Loss);
            metricsLog.ValidationLoss.Add(validationMetrics.Loss);
            metricsLog.TrainMetrics.Add(trainingMetrics.Metric);
            metricsLog.ValidationMetrics.Add(validationMetrics.Metric);
            metricsLog.EpochSeconds.Add((float)epochTimer.Elapsed.TotalSeconds);

            File.WriteAllText(metricsPath, JsonSerializer.Serialize(metricsLog, JsonOptions));

            SaveModelCheckpoint(
                Path.Combine(checkpointsPath, $"model-epoch-{epoch:0000}.pt"),
                model,
                trainingState,
                new EpochSummary(
                    epoch,
                    maxEpochs,
                    trainingMetrics.Loss,
                    validationMetrics.Loss,
                    trainingMetrics.Metric,
                    validationMetrics.Metric,
                    epochTimer.Elapsed.TotalSeconds));

            if (epoch % printEveryEpoch == 0 || epoch == 1 || epoch == maxEpochs)
            {
                Console.WriteLine(
                    $"Epoch {epoch}/{maxEpochs} | train loss: {trainingMetrics.Loss:F6} | " +
                    $"val loss: {validationMetrics.Loss:F6}");
            }
        }

        var testingMetrics = RunStubEpoch(model, optimizer: null, batchSize: Math.Max(1, _settings.EvaluationBatchSize), training: false);
        metricsLog.TestLoss.Add(testingMetrics.Loss);
        metricsLog.TestMetrics.Add(testingMetrics.Metric);
        File.WriteAllText(metricsPath, JsonSerializer.Serialize(metricsLog, JsonOptions));

        var finalReport =
            new FinalTrainingReport(trainingState.CompletedEpochs, testingMetrics.Loss, testingMetrics.Metric);

        SaveFinalMetricsAsJson(finalReport, runDirectoryPath);
        Console.WriteLine(
            $"Training finished. Epochs: {finalReport.CompletedEpochs}, " +
            $"test loss: {finalReport.TestLoss:F6}, test metric: {finalReport.TestMetric:F6}");

        return Task.CompletedTask;
    }

    private static StubTextToMotionModel CreateModel() => new();

    private static StubEpochMetrics RunStubEpoch(
        StubTextToMotionModel model,
        torch.optim.Optimizer? optimizer,
        int batchSize,
        bool training)
    {
        using var scope = torch.NewDisposeScope();

        var inputs = randn([batchSize, StubTextToMotionModel.InputFeatures]);
        var targets = zeros([batchSize, StubTextToMotionModel.OutputFeatures]);
        var predictions = model.forward(inputs);
        var loss = functional.mse_loss(predictions, targets);

        if (training && optimizer is not null)
        {
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        float lossValue = loss.ToSingle();
        return new StubEpochMetrics(lossValue, 1.0f / (1.0f + lossValue));
    }

    private static void SaveModelCheckpoint(
        string checkpointPath,
        StubTextToMotionModel model,
        TrainingState trainingState,
        EpochSummary epochSummary)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(checkpointPath)!);

        var checkpoint = new StubCheckpoint(
            ModelName: model.GetType().Name,
            Epoch: epochSummary.Epoch,
            CompletedEpochs: trainingState.CompletedEpochs,
            LearningRate: trainingState.LatestLearningRate,
            Summary: epochSummary,
            CreatedUtc: DateTime.UtcNow);

        File.WriteAllText(checkpointPath, JsonSerializer.Serialize(checkpoint, JsonOptions));
    }

    private static void SaveFinalMetricsAsJson(FinalTrainingReport report, string runDirectoryPath)
    {
        string reportPath = Path.Combine(runDirectoryPath, "final-report.json");
        File.WriteAllText(reportPath, JsonSerializer.Serialize(report, JsonOptions));
    }

    private static string InitializeRunDirectory(ModelSettings settings)
    {
        string runsRootPath = string.IsNullOrWhiteSpace(settings.RunsRootPath)
            ? Path.Combine(AppContext.BaseDirectory, "Runs")
            : Path.GetFullPath(settings.RunsRootPath);

        Directory.CreateDirectory(runsRootPath);

        int nextRunNumber = Directory.GetDirectories(runsRootPath, "Trainer-*")
            .Select(Path.GetFileName)
            .Select(name =>
            {
                if (string.IsNullOrWhiteSpace(name))
                    return 0;

                string[] parts = name.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                return parts.Length >= 2 && int.TryParse(parts[^1], out int number) ? number : 0;
            })
            .DefaultIfEmpty(0)
            .Max() + 1;

        string runDirectoryPath = Path.Combine(runsRootPath, $"Trainer-{nextRunNumber:000}");
        Directory.CreateDirectory(runDirectoryPath);
        return runDirectoryPath;
    }

    private static void SetRandomSeed(int seed)
    {
        long normalizedSeed = Math.Max(0, seed);
        torch.random.manual_seed(normalizedSeed);
        if (cuda.is_available())
            torch.cuda.manual_seed(normalizedSeed);
    }

    private sealed class TrainingState
    {
        public int CompletedEpochs { get; set; }
        public float LatestLearningRate { get; set; }
    }

    private sealed record StubEpochMetrics(float Loss, float Metric);

    private sealed record EpochSummary(int Epoch, int MaxEpochs, float TrainLoss, float ValidationLoss,
        float TrainMetric, float ValidationMetric, double EpochSeconds);

    private sealed record FinalTrainingReport(int CompletedEpochs, float TestLoss, float TestMetric);

    private sealed record StubCheckpoint(string ModelName, int Epoch, int CompletedEpochs, float LearningRate,
        EpochSummary Summary, DateTime CreatedUtc);
}
