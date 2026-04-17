using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Options;

namespace Text2Motion.TorchTrainer;

public class TextToMotionModelTrainer(IOptions<ModelSettings> options)
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly ModelSettings _settings = options.Value;

    public async Task TrainAsync(CancellationToken token)
    {
        int maxEpochs = _settings.MaxEpochs;
        int printEveryEpoch = _settings.PrintEveryEpoch;

        SetRandomSeed(_settings.RandomSeed);
        string device = _settings.Device;

        string runDirectoryPath = InitializeRunDirectory(_settings);
        string checkpointsPath = Path.Combine(runDirectoryPath, "checkpoints");
        string metricsPath = Path.Combine(runDirectoryPath, "metrics.json");
        string settingsPath = Path.Combine(runDirectoryPath, "model-settings.json");

        Directory.CreateDirectory(checkpointsPath);
        File.WriteAllText(settingsPath, JsonSerializer.Serialize(_settings, JsonOptions));

        var normalizationStats = LoadNormalizationStatistics(_settings);
        var datasets = BuildDatasets(_settings, normalizationStats);
        var metricsLog = new TrainerMetricsLog();

        var trainLoader = CreateMiniBatchDataLoader(datasets.Train, _settings, shuffle: true);
        var validationLoader = CreateMiniBatchDataLoader(datasets.Validation, _settings, shuffle: false);

        var model = CreateModel(_settings).to(device);
        var optimizer = CreateOptimizer(model, _settings);
        var scheduler = CreateLearningRateScheduler(optimizer, _settings);
        var gradientScaler = CreateGradientScaler(_settings);
        var trainingState = InitializeTrainingState(_settings);

        if (ShouldResumeTraining(_settings))
            RestoreTrainingState(_settings, model, optimizer, scheduler, gradientScaler, trainingState, metricsLog);

        for (int epoch = trainingState.StartEpoch; epoch <= maxEpochs; epoch++)
        {
            token.ThrowIfCancellationRequested();

            var epochTimer = Stopwatch.StartNew();
            Console.WriteLine($"Epoch {epoch}/{maxEpochs} started.");

            model.train();
            var trainingMetrics = ProcessTrainingLoop(model, trainLoader, optimizer, scheduler, gradientScaler,
                trainAccumulator, device, _settings, normalizationStats, token);

            model.eval();
            var validationMetrics = ProcessValidationLoop(model, validationLoader, device,
                _settings, normalizationStats, token);

            epochTimer.Stop();

            var epochSummary = CreateEpochSummary(epoch, maxEpochs, trainingMetrics, validationMetrics, 
                epochTimer.Elapsed, optimizer, scheduler, trainingState);

            metricsLog.Epochs.Add(epoch);
            metricsLog.TrainLoss.Add(trainingMetrics.Loss);
            metricsLog.ValidationLoss.Add(validationMetrics.Loss);
            metricsLog.TrainMetrics.Add(trainingMetrics.Metric);
            metricsLog.ValidationMetrics.Add(validationMetrics.Metric);
            metricsLog.EpochSeconds.Add((float)epochTimer.Elapsed.TotalSeconds);

            File.WriteAllText(metricsPath, JsonSerializer.Serialize(metricsLog, JsonOptions));

            SaveModelCheckpoint(Path.Combine(checkpointsPath, $"model-epoch-{epoch:0000}.pt"), model, optimizer,
                scheduler, gradientScaler, trainingState, epochSummary);

            if (epoch % printEveryEpoch == 0 || epoch == 1 || epoch == maxEpochs)
                DisplayEpochModelState(epochSummary, printEveryEpoch);
        }

        // Use the same model instance

        var testLoader = CreateMiniBatchDataLoader(datasets.Test, _settings, shuffle: false);
        model.eval();
        var testingMetrics = ProcessEvaluationLoop(model, testLoader, device, _settings,
            normalizationStats, token);

        metricsLog.TestLoss.Add(testingMetrics.Loss);
        metricsLog.TestMetrics.Add(testingMetrics.Metric);
        File.WriteAllText(metricsPath, JsonSerializer.Serialize(metricsLog, JsonOptions));

        var finalTrainingReport = CreateFinalTrainingReport(trainingState, testingMetrics, metricsLog, maxEpochs);
        SaveFinalMetricsAsJson(finalTrainingReport, _settings);
        DisplayFinalTrainingState(finalTrainingReport);
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
}