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
        string checkpointsPath = Path.Combine(outputRootPath, "checkpoints");
        string resultsPath = Path.Combine(outputRootPath, "results");
        string metricsPath = Path.Combine(resultsPath, "metrics.json");
        string modelSettingsSnapshotPath = Path.Combine(outputRootPath, "model-settings.snapshot.json");
        string trainingSettingsSnapshotPath = Path.Combine(outputRootPath, "training-settings.snapshot.json");

        Directory.CreateDirectory(outputRootPath);
        Directory.CreateDirectory(checkpointsPath);
        Directory.CreateDirectory(resultsPath);

        File.WriteAllText(modelSettingsSnapshotPath, JsonSerializer.Serialize(_modelSettings, JsonOptions));
        File.WriteAllText(trainingSettingsSnapshotPath, JsonSerializer.Serialize(_trainingSettings, JsonOptions));

        var metricsLog = new TrainerMetricsLog();
        var trainingState = new TrainingState();
        var model = CreateModel();
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
            SaveEpochArtifacts(checkpointsPath, resultsPath, model, trainingState, epochSummary);

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

        var finalReport = new FinalTrainingReport(
            trainingState.CompletedEpochs,
            testingMetrics.Loss,
            testingMetrics.Metric);

        SaveFinalArtifacts(outputRootPath, resultsPath, model, finalReport);

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

    private static void SaveEpochArtifacts(
        string checkpointsPath,
        string resultsPath,
        StubTextToMotionModel model,
        TrainingState trainingState,
        EpochSummary epochSummary)
    {
        string weightsPath = Path.Combine(checkpointsPath, $"model-epoch-{epochSummary.Epoch:0000}.pt");
        string checkpointInfoPath = Path.Combine(resultsPath, $"model-epoch-{epochSummary.Epoch:0000}.json");

        model.save(weightsPath);

        var checkpoint = new StubCheckpoint(
            ModelName: model.GetType().Name,
            WeightsPath: weightsPath,
            Epoch: epochSummary.Epoch,
            CompletedEpochs: trainingState.CompletedEpochs,
            LearningRate: trainingState.LatestLearningRate,
            Summary: epochSummary,
            CreatedUtc: DateTime.UtcNow);

        File.WriteAllText(checkpointInfoPath, JsonSerializer.Serialize(checkpoint, JsonOptions));
    }

    private static void SaveFinalArtifacts(
        string outputRootPath,
        string resultsPath,
        StubTextToMotionModel model,
        FinalTrainingReport report)
    {
        string finalWeightsPath = Path.Combine(outputRootPath, "model-final.pt");
        string finalReportPath = Path.Combine(resultsPath, "final-report.json");

        model.save(finalWeightsPath);
        File.WriteAllText(finalReportPath, JsonSerializer.Serialize(report, JsonOptions));
    }

    private static string ResolveOutputRootPath(TrainingSettings settings)
    {
        string outputRootPath = string.IsNullOrWhiteSpace(settings.OutputRootPath)
            ? Path.Combine(AppContext.BaseDirectory, "Weights", "Text2Motion")
            : Path.GetFullPath(settings.OutputRootPath);

        Directory.CreateDirectory(outputRootPath);
        return outputRootPath;
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

    private sealed record FinalTrainingReport(
        int CompletedEpochs,
        float TestLoss,
        float TestMetric);

    private sealed record StubCheckpoint(
        string ModelName,
        string WeightsPath,
        int Epoch,
        int CompletedEpochs,
        float LearningRate,
        EpochSummary Summary,
        DateTime CreatedUtc);
}
