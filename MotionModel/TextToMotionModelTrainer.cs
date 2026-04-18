using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
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

    private static readonly Regex ModelCheckpointRegex = new(@"^model-epoch-(\d{4})\.pt$", RegexOptions.Compiled);

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
        var optimizer = torch.optim.Adam(
            model.parameters(),
            lr: _modelSettings.LearningRate,
            weight_decay: _modelSettings.WeightDecay);

        int startEpoch = 1;
        if (_trainingSettings.LoadCheckpoint)
            startEpoch = RestoreCheckpoint(runDirectoryPath, model, optimizer, metricsLog) + 1;

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
            SaveEpochCheckpoint(checkpointsPath, model, optimizer, epoch);

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
        SaveFinalArtifacts(runDirectoryPath, testMetricsPath, model, optimizer, testMetricsLog);

        Console.WriteLine(
            $"Training finished. Epochs: {metricsLog.Epochs.LastOrDefault()}, " +
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

    private static void SaveEpochCheckpoint(
        string checkpointsPath,
        StubTextToMotionModel model,
        torch.optim.Optimizer optimizer,
        int epoch)
    {
        model.save(Path.Combine(checkpointsPath, $"model-epoch-{epoch:0000}.pt"));
        SaveOptimizerMetadata(Path.Combine(checkpointsPath, $"optimizer-epoch-{epoch:0000}.json"), optimizer);
    }

    private static void SaveFinalArtifacts(
        string runDirectoryPath,
        string testMetricsPath,
        StubTextToMotionModel model,
        torch.optim.Optimizer optimizer,
        TrainerMetricsLog testMetrics)
    {
        model.save(Path.Combine(runDirectoryPath, "model-final.pt"));
        SaveOptimizerMetadata(Path.Combine(runDirectoryPath, "optimizer-final.json"), optimizer);
        File.WriteAllText(testMetricsPath, JsonSerializer.Serialize(testMetrics, JsonOptions));
    }

    private static int RestoreCheckpoint(
        string runDirectoryPath,
        StubTextToMotionModel model,
        torch.optim.Optimizer optimizer,
        TrainerMetricsLog metricsLog)
    {
        string checkpointsPath = Path.Combine(runDirectoryPath, "checkpoints");
        if (!Directory.Exists(checkpointsPath))
            throw new InvalidOperationException($"Checkpoint directory does not exist: {checkpointsPath}");

        int latestEpoch = Directory.GetFiles(checkpointsPath, "model-epoch-*.pt")
            .Select(Path.GetFileName)
            .Select(TryParseCheckpointEpoch)
            .Where(epoch => epoch.HasValue)
            .Select(epoch => epoch!.Value)
            .DefaultIfEmpty(0)
            .Max();

        if (latestEpoch == 0)
            throw new InvalidOperationException($"No model checkpoints found in {checkpointsPath}");

        string modelCheckpointPath = Path.Combine(checkpointsPath, $"model-epoch-{latestEpoch:0000}.pt");
        string optimizerCheckpointPath = Path.Combine(checkpointsPath, $"optimizer-epoch-{latestEpoch:0000}.json");

        if (!File.Exists(optimizerCheckpointPath))
            throw new InvalidOperationException($"Optimizer checkpoint does not exist: {optimizerCheckpointPath}");

        model.load(modelCheckpointPath);
        LoadOptimizerMetadata(optimizerCheckpointPath);

        TrimMetricsToEpoch(metricsLog, latestEpoch);

        Console.WriteLine($"Resumed training from Run-{Path.GetFileName(runDirectoryPath)?.Split('-').Last()} epoch {latestEpoch:0000}.");
        return latestEpoch;
    }

    private static void SaveOptimizerMetadata(string optimizerPath, torch.optim.Optimizer optimizer)
    {
        var metadata = new OptimizerMetadata(
            OptimizerType: optimizer.GetType().Name);

        File.WriteAllText(optimizerPath, JsonSerializer.Serialize(metadata, JsonOptions));
    }

    private static void LoadOptimizerMetadata(string optimizerPath)
    {
        if (!File.Exists(optimizerPath))
            throw new InvalidOperationException($"Optimizer checkpoint does not exist: {optimizerPath}");

        _ = JsonSerializer.Deserialize<OptimizerMetadata>(File.ReadAllText(optimizerPath), JsonOptions)
            ?? throw new InvalidOperationException($"Optimizer checkpoint is invalid: {optimizerPath}");
    }

    private static void TrimMetricsToEpoch(TrainerMetricsLog metricsLog, int epochCount)
    {
        TrimList(metricsLog.Epochs, epochCount);
        TrimList(metricsLog.TrainLoss, epochCount);
        TrimList(metricsLog.ValidationLoss, epochCount);
        TrimList(metricsLog.TrainMetrics, epochCount);
        TrimList(metricsLog.ValidationMetrics, epochCount);
        TrimList(metricsLog.EpochSeconds, epochCount);
    }

    private static void TrimList<T>(List<T> values, int count)
    {
        if (values.Count > count)
            values.RemoveRange(count, values.Count - count);
    }

    private static int? TryParseCheckpointEpoch(string? fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            return null;

        var match = ModelCheckpointRegex.Match(fileName);
        return match.Success && int.TryParse(match.Groups[1].Value, out int epoch) ? epoch : null;
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
        torch.random.manual_seed(normalizedSeed);
        if (cuda.is_available())
            torch.cuda.manual_seed(normalizedSeed);
    }

    private sealed record StubEpochMetrics(float Loss, float Metric);

    private sealed record OptimizerMetadata(string OptimizerType);
}
