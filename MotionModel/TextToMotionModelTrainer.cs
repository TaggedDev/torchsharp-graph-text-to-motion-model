using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Options;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class TextToMotionModelTrainer(IOptions<ModelSettings> modelOptions, IOptions<TrainingSettings> trainingOptions)
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly ModelSettings _modelSettings = modelOptions.Value;
    private readonly TrainingSettings _trainingSettings = trainingOptions.Value;

    public Task TrainAsync(CancellationToken token)
    {
        int maxEpochs = Math.Max(1, _modelSettings.MaxEpochs);
        int printEveryEpoch = Math.Max(1, _modelSettings.PrintEveryEpoch);

        SetRandomSeed(_modelSettings.RandomSeed);

        string outputRootPath = ResolveOutputRootPath(_trainingSettings);
        string runDirectoryPath = CreateRunDirectory(outputRootPath);
        string checkpointsPath = Path.Combine(runDirectoryPath, "checkpoints");
        string resultsPath = Path.Combine(runDirectoryPath, "results");
        string metricsPath = Path.Combine(resultsPath, "metrics.json");
        string testMetricsPath = Path.Combine(resultsPath, "test-metrics.json");
        string modelSettingsSnapshotPath = Path.Combine(runDirectoryPath, "model-settings.snapshot.json");
        string trainingSettingsSnapshotPath = Path.Combine(runDirectoryPath, "training-settings.snapshot.json");

        Directory.CreateDirectory(runDirectoryPath);
        Directory.CreateDirectory(checkpointsPath);
        Directory.CreateDirectory(resultsPath);

        File.WriteAllText(modelSettingsSnapshotPath, JsonSerializer.Serialize(_modelSettings, JsonOptions));
        File.WriteAllText(trainingSettingsSnapshotPath, JsonSerializer.Serialize(_trainingSettings, JsonOptions));

        var metricsLog = new TrainerMetricsLog();
        var trainingState = new TrainingState();
        var model = new StubTextToMotionModel();
        var optimizer = torch.optim.Adam(
            model.parameters(),
            lr: _modelSettings.LearningRate,
            weight_decay: _modelSettings.WeightDecay);

        for (int epoch = 1; epoch <= maxEpochs; epoch++)
        {
            token.ThrowIfCancellationRequested();

            var epochTimer = Stopwatch.StartNew();
            Console.WriteLine($"Epoch {epoch}/{maxEpochs} started.");

            model.train();
            var trainingMetrics = RunStubEpoch(
                model,
                optimizer,
                batchSize: Math.Max(1, _modelSettings.BatchSize),
                training: true);

            model.eval();
            var validationMetrics = RunStubEpoch(
                model,
                optimizer: null,
                batchSize: Math.Max(1, _modelSettings.EvaluationBatchSize),
                training: false);

            epochTimer.Stop();

            trainingState.CompletedEpochs = epoch;
            trainingState.LatestLearningRate = _modelSettings.LearningRate;

            var epochSummary = new EpochSummary(
                epoch,
                maxEpochs,
                trainingMetrics.Loss,
                validationMetrics.Loss,
                trainingMetrics.Metric,
                validationMetrics.Metric,
                epochTimer.Elapsed.TotalSeconds);

            metricsLog.Epochs.Add(epoch);
            metricsLog.TrainLoss.Add(trainingMetrics.Loss);
            metricsLog.ValidationLoss.Add(validationMetrics.Loss);
            metricsLog.TrainMetrics.Add(trainingMetrics.Metric);
            metricsLog.ValidationMetrics.Add(validationMetrics.Metric);
            metricsLog.EpochSeconds.Add((float)epochTimer.Elapsed.TotalSeconds);

            File.WriteAllText(metricsPath, JsonSerializer.Serialize(metricsLog, JsonOptions));
            SaveEpochWeights(checkpointsPath, model, epochSummary.Epoch);

            if (epoch % printEveryEpoch == 0 || epoch == 1 || epoch == maxEpochs)
            {
                Console.WriteLine(
                    $"Epoch {epoch}/{maxEpochs} | train loss: {trainingMetrics.Loss:F6} | " +
                    $"val loss: {validationMetrics.Loss:F6}");
            }
        }

        var testingMetrics = RunStubEpoch(
            model,
            optimizer: null,
            batchSize: Math.Max(1, _modelSettings.EvaluationBatchSize),
            training: false);

        metricsLog.TestLoss.Add(testingMetrics.Loss);
        metricsLog.TestMetrics.Add(testingMetrics.Metric);
        File.WriteAllText(metricsPath, JsonSerializer.Serialize(metricsLog, JsonOptions));

        var testMetricsLog = CreateTestMetricsLog(trainingState.CompletedEpochs, testingMetrics);
        SaveFinalArtifacts(runDirectoryPath, testMetricsPath, model, testMetricsLog);

        Console.WriteLine(
            $"Training finished. Epochs: {trainingState.CompletedEpochs}, " +
            $"test loss: {testingMetrics.Loss:F6}, test metric: {testingMetrics.Metric:F6}");

        return Task.CompletedTask;
    }

    private static StubEpochMetrics RunStubEpoch(
        StubTextToMotionModel model,
        torch.optim.Optimizer? optimizer,
        int batchSize,
        bool training)
    {
        using var scope = NewDisposeScope();

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

    private static void SaveEpochWeights(
        string checkpointsPath,
        StubTextToMotionModel model,
        int epoch)
    {
        string weightsPath = Path.Combine(checkpointsPath, $"model-epoch-{epoch:0000}.pt");
        model.save(weightsPath);
    }

    private static void SaveFinalArtifacts(
        string runDirectoryPath,
        string testMetricsPath,
        StubTextToMotionModel model,
        TrainerMetricsLog testMetrics)
    {
        string finalWeightsPath = Path.Combine(runDirectoryPath, "model-final.pt");

        model.save(finalWeightsPath);
        File.WriteAllText(testMetricsPath, JsonSerializer.Serialize(testMetrics, JsonOptions));
    }

    private static string ResolveOutputRootPath(TrainingSettings settings)
    {
        string outputRootPath = string.IsNullOrWhiteSpace(settings.OutputRootPath)
            ? Path.Combine(AppContext.BaseDirectory, "Weights", "Text2Motion")
            : Path.GetFullPath(settings.OutputRootPath);

        Directory.CreateDirectory(outputRootPath);
        return outputRootPath;
    }

    private static string CreateRunDirectory(string outputRootPath)
    {
        int nextRunNumber = Directory.GetDirectories(outputRootPath, "Run-*")
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

        string runDirectoryPath = Path.Combine(outputRootPath, $"Run-{nextRunNumber:0000}");
        Directory.CreateDirectory(runDirectoryPath);
        return runDirectoryPath;
    }

    private static TrainerMetricsLog CreateTestMetricsLog(int completedEpochs, StubEpochMetrics testingMetrics)
    {
        return new TrainerMetricsLog
        {
            Epochs = [completedEpochs],
            TestLoss = [testingMetrics.Loss],
            TestMetrics = [testingMetrics.Metric],
            EpochSeconds = [0f]
        };
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

    private sealed record EpochSummary(
        int Epoch,
        int MaxEpochs,
        float TrainLoss,
        float ValidationLoss,
        float TrainMetric,
        float ValidationMetric,
        double EpochSeconds);
    
}
