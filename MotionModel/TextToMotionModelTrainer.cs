using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Options;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class TextToMotionModelTrainer(
    IOptions<ModelSettings> modelOptions,
    IOptions<TrainingSettings> trainingOptions,
    ModelCheckpointService checkpointService)
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
        string runDirectoryPath = ResolveRunDirectory(outputRootPath, _trainingSettings);
        string checkpointsPath = Path.Combine(runDirectoryPath, "checkpoints");
        string resultsPath = Path.Combine(runDirectoryPath, "results");
        string metricsPath = Path.Combine(resultsPath, "metrics.json");
        string testMetricsPath = Path.Combine(resultsPath, "test-metrics.json");

        Directory.CreateDirectory(runDirectoryPath);
        Directory.CreateDirectory(checkpointsPath);
        Directory.CreateDirectory(resultsPath);

        var metricsLog = LoadOrCreateMetricsLog(metricsPath, _trainingSettings.LoadCheckpoint);
        var model = new StubTextToMotionModel();
        var optimizer = optim.Adam(
            model.parameters(),
            lr: _modelSettings.LearningRate,
            weight_decay: _modelSettings.WeightDecay);

        int startEpoch = 1;
        if (_trainingSettings.LoadCheckpoint)
            startEpoch = checkpointService.RestoreCheckpoint(runDirectoryPath, model, metricsLog) + 1;

        if (startEpoch > maxEpochs)
        {
            Console.WriteLine(
                $"No training steps executed. Resume epoch {startEpoch} is greater than configured max epoch {maxEpochs}.");
            return Task.CompletedTask;
        }

        for (int epoch = startEpoch; epoch <= maxEpochs; epoch++)
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

            metricsLog.Epochs.Add(epoch);
            metricsLog.TrainLoss.Add(trainingMetrics.Loss);
            metricsLog.ValidationLoss.Add(validationMetrics.Loss);
            metricsLog.TrainMetrics.Add(trainingMetrics.Metric);
            metricsLog.ValidationMetrics.Add(validationMetrics.Metric);
            metricsLog.EpochSeconds.Add((float)epochTimer.Elapsed.TotalSeconds);

            File.WriteAllText(metricsPath, JsonSerializer.Serialize(metricsLog, JsonOptions));
            checkpointService.SaveEpochCheckpoint(checkpointsPath, model, epoch);

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

        var testMetricsLog = CreateTestMetricsLog(metricsLog.Epochs.LastOrDefault(), testingMetrics);
        checkpointService.SaveFinalArtifacts(runDirectoryPath, testMetricsPath, model, testMetricsLog);

        Console.WriteLine(
            $"Training finished. Epochs: {metricsLog.Epochs.LastOrDefault()}, " +
            $"test loss: {testingMetrics.Loss:F6}, test metric: {testingMetrics.Metric:F6}");

        return Task.CompletedTask;
    }

    private static StubEpochMetrics RunStubEpoch(
        StubTextToMotionModel model,
        optim.Optimizer? optimizer,
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

    private static string ResolveOutputRootPath(TrainingSettings settings)
    {
        string outputRootPath = string.IsNullOrWhiteSpace(settings.OutputRootPath)
            ? Path.Combine(AppContext.BaseDirectory, "Weights", "Text2Motion")
            : Path.GetFullPath(settings.OutputRootPath);

        Directory.CreateDirectory(outputRootPath);
        return outputRootPath;
    }

    private static string ResolveRunDirectory(string outputRootPath, TrainingSettings settings)
    {
        if (settings.LoadCheckpoint)
        {
            if (settings.LoadRunNumber <= 0)
                throw new InvalidOperationException("Training.LoadRunNumber must be greater than 0 when Training.LoadCheckpoint is true.");

            string existingRunPath = Path.Combine(outputRootPath, $"Run-{settings.LoadRunNumber:0000}");
            if (!Directory.Exists(existingRunPath))
                throw new InvalidOperationException($"Run directory does not exist: {existingRunPath}");

            return existingRunPath;
        }

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

    private static TrainerMetricsLog LoadOrCreateMetricsLog(string metricsPath, bool loadCheckpoint)
    {
        if (!loadCheckpoint || !File.Exists(metricsPath))
            return new TrainerMetricsLog();

        return JsonSerializer.Deserialize<TrainerMetricsLog>(File.ReadAllText(metricsPath), JsonOptions)
               ?? new TrainerMetricsLog();
    }

    private static TrainerMetricsLog CreateTestMetricsLog(int completedEpochs, StubEpochMetrics testingMetrics)
    {
        return new TrainerMetricsLog
        {
            Epochs = [completedEpochs],
            TestLoss = [testingMetrics.Loss],
            TestMetrics = [testingMetrics.Metric]
        };
    }

    private static void SetRandomSeed(int seed)
    {
        long normalizedSeed = Math.Max(0, seed);
        random.manual_seed(normalizedSeed);
        if (cuda.is_available())
            cuda.manual_seed(normalizedSeed);
    }

    private sealed record StubEpochMetrics(float Loss, float Metric);
}
