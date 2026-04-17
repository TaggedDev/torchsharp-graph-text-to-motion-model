using System.Collections;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.Options;

namespace Text2Motion.TorchTrainer;

public class TextToMotionModelTrainer(IOptions<ModelSettings> options)
{
    private const string Validation = "validation";
    private const string Train = "train";
    private const string Test = "test";

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

        var runContext = InitializeTrainerRun(_settings);
        SaveModelSettings(runContext.SettingsSnapshotPath, _settings);

        var normalizationStats = LoadNormalizationStatistics(_settings);
        var datasets = BuildDatasets(_settings, normalizationStats);
        var metricHistory = LoadOrCreateMetricHistory(runContext.MetricsPath);

        var trainLoader = CreateMiniBatchDataLoader(datasets.Train, _settings, shuffle: true);
        var validationLoader = CreateMiniBatchDataLoader(datasets.Validation, _settings, shuffle: false);

        var model = CreateModel(_settings).to(device);
        var optimizer = CreateOptimizer(model, _settings);
        var scheduler = CreateLearningRateScheduler(optimizer, _settings);
        var gradientScaler = CreateGradientScaler(_settings);
        var trainingState = InitializeTrainingState(_settings);

        if (ShouldResumeTraining(_settings))
            RestoreTrainingState(_settings, model, optimizer, scheduler, gradientScaler, trainingState, metricHistory);

        for (int epoch = trainingState.StartEpoch; epoch <= maxEpochs; epoch++)
        {
            token.ThrowIfCancellationRequested();

            var epochTimer = StartEpochTimer();
            Console.WriteLine($"Epoch {epoch}/{maxEpochs} started.");

            var trainAccumulator = CreateMetricAccumulator(Train);
            model.train();
            var trainingMetrics = ProcessTrainingLoop(model, trainLoader, optimizer, scheduler, gradientScaler,
                trainAccumulator, device, _settings, normalizationStats, token);

            var validationAccumulator = CreateMetricAccumulator(Validation);
            model.eval();
            var validationMetrics = ProcessValidationLoop(model, validationLoader, validationAccumulator, device,
                _settings, normalizationStats, token);

            var epochDuration = CompleteEpochTimer(epochTimer);
            var epochSummary = CreateEpochSummary(epoch, maxEpochs, trainingMetrics, validationMetrics, epochDuration,
                optimizer, scheduler, trainingState);

            HandleMetadataAfterEpoch(runContext, metricHistory, epochSummary, trainingMetrics, validationMetrics,
                trainingState, _settings, model, optimizer, scheduler, gradientScaler, epoch, printEveryEpoch,
                maxEpochs);
        }

        LoadBestModelStateForFinalEvaluation(model, _settings, device);

        var testLoader = CreateMiniBatchDataLoader(datasets.Test, _settings, shuffle: false);
        var testAccumulator = CreateMetricAccumulator(Test);
        model.eval();
        var testingMetrics = ProcessEvaluationLoop(model, testLoader, testAccumulator, device, _settings,
            normalizationStats, token);

        AppendPhaseMetrics(metricHistory, $"{Test}.", testingMetrics);
        SaveMetricHistory(runContext.MetricsPath, metricHistory);

        var finalTrainingReport = CreateFinalTrainingReport(trainingState, testingMetrics, metricHistory, maxEpochs);
        SaveFinalMetricsAsJson(finalTrainingReport, _settings);
        DisplayFinalTrainingState(finalTrainingReport);
    }

    private static void HandleMetadataAfterEpoch(TrainerRunContext runContext, MetricHistory metricHistory,
        object epochSummary, object trainingMetrics, object validationMetrics, object trainingState, object settings,
        object model, object optimizer, object scheduler, object gradientScaler, int epoch, int printEveryEpoch,
        int maxEpochs)
    {
        AppendEpochIndex(metricHistory, epoch);
        AppendPhaseMetrics(metricHistory, $"{Train}.", trainingMetrics);
        AppendPhaseMetrics(metricHistory, $"{Validation}.", validationMetrics);
        AppendEpochSummaryMetrics(metricHistory, epochSummary);
        SaveMetricHistory(runContext.MetricsPath, metricHistory);

        UpdateBestModelState(trainingState, epochSummary);
        UpdateEarlyStoppingState(trainingState, epochSummary, settings);

        SaveModelState(runContext, model, optimizer, scheduler, gradientScaler, trainingState, epochSummary, epoch);
        SaveBestModelStateIfNeeded(runContext, model, optimizer, scheduler, gradientScaler, trainingState,
            epochSummary, epoch);

        if (epoch % printEveryEpoch == 0 || epoch == 1 || epoch == maxEpochs)
            DisplayEpochModelState(epochSummary, printEveryEpoch);
    }

    private static Stopwatch StartEpochTimer()
        => Stopwatch.StartNew();

    private static TimeSpan CompleteEpochTimer(Stopwatch epochTimer)
    {
        epochTimer.Stop();
        return epochTimer.Elapsed;
    }

    private static TrainerRunContext InitializeTrainerRun(ModelSettings settings)
    {
        string runsRootPath = string.IsNullOrWhiteSpace(settings.RunsRootPath)
            ? Path.Combine(AppContext.BaseDirectory, "Runs")
            : Path.GetFullPath(settings.RunsRootPath);

        Directory.CreateDirectory(runsRootPath);

        int nextRunNumber = Directory.GetDirectories(runsRootPath, "Trainer-*")
            .Select(Path.GetFileName)
            .Select(GetRunNumberOrDefault)
            .DefaultIfEmpty(0)
            .Max() + 1;

        string runDirectoryName = $"Trainer-{nextRunNumber:000}";
        string runDirectoryPath = Path.Combine(runsRootPath, runDirectoryName);
        string checkpointsPath = Path.Combine(runDirectoryPath, "checkpoints");

        Directory.CreateDirectory(runDirectoryPath);
        Directory.CreateDirectory(checkpointsPath);

        return new TrainerRunContext(
            runDirectoryPath,
            checkpointsPath,
            Path.Combine(runDirectoryPath, "model-settings.json"),
            Path.Combine(runDirectoryPath, "metrics.json"));
    }

    private static int GetRunNumberOrDefault(string? directoryName)
    {
        if (string.IsNullOrWhiteSpace(directoryName))
            return 0;

        string[] parts = directoryName.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 2)
            return 0;

        return int.TryParse(parts[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int runNumber)
            ? runNumber
            : 0;
    }

    private static void SaveModelSettings(string settingsSnapshotPath, ModelSettings settings)
    {
        string json = JsonSerializer.Serialize(settings, JsonOptions);
        File.WriteAllText(settingsSnapshotPath, json);
    }

    private static MetricHistory LoadOrCreateMetricHistory(string metricsPath)
    {
        if (!File.Exists(metricsPath))
            return new MetricHistory();

        string json = File.ReadAllText(metricsPath);
        return JsonSerializer.Deserialize<MetricHistory>(json, JsonOptions) ?? new MetricHistory();
    }

    private static void SaveMetricHistory(string metricsPath, MetricHistory metricHistory)
    {
        string json = JsonSerializer.Serialize(metricHistory, JsonOptions);
        File.WriteAllText(metricsPath, json);
    }

    private static void AppendEpochIndex(MetricHistory metricHistory, int epoch)
    {
        metricHistory.Epochs.Add(epoch);
    }

    private static void AppendEpochSummaryMetrics(MetricHistory metricHistory, object epochSummary)
    {
        foreach (var (metricName, metricValue) in FlattenNumericMetrics(epochSummary))
            AppendMetric(metricHistory, metricName, metricValue);
    }

    private static void AppendPhaseMetrics(MetricHistory metricHistory, string prefix, object phaseMetrics)
    {
        foreach (var (metricName, metricValue) in FlattenNumericMetrics(phaseMetrics))
            AppendMetric(metricHistory, $"{prefix}{metricName}", metricValue);
    }

    private static void AppendMetric(MetricHistory metricHistory, string metricName, float metricValue)
    {
        if (!metricHistory.Metrics.TryGetValue(metricName, out var values))
        {
            values = [];
            metricHistory.Metrics[metricName] = values;
        }

        values.Add(metricValue);
    }

    private static IEnumerable<(string Name, float Value)> FlattenNumericMetrics(object value, string prefix = "")
    {
        if (value is null)
            yield break;

        if (TryConvertToFloat(value, out float numericValue))
        {
            yield return (NormalizeMetricName(prefix), numericValue);
            yield break;
        }

        if (value is IDictionary dictionary)
        {
            foreach (DictionaryEntry entry in dictionary)
            {
                string childName = BuildMetricName(prefix, entry.Key?.ToString());
                foreach (var metric in FlattenNumericMetrics(entry.Value!, childName))
                    yield return metric;
            }

            yield break;
        }

        var properties = value.GetType()
            .GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(property => property.CanRead && property.GetIndexParameters().Length == 0);

        foreach (var property in properties)
        {
            object? propertyValue = property.GetValue(value);
            string childName = BuildMetricName(prefix, property.Name);
            foreach (var metric in FlattenNumericMetrics(propertyValue!, childName))
                yield return metric;
        }
    }

    private static bool TryConvertToFloat(object value, out float numericValue)
    {
        switch (value)
        {
            case byte byteValue:
                numericValue = byteValue;
                return true;
            case sbyte sbyteValue:
                numericValue = sbyteValue;
                return true;
            case short shortValue:
                numericValue = shortValue;
                return true;
            case ushort ushortValue:
                numericValue = ushortValue;
                return true;
            case int intValue:
                numericValue = intValue;
                return true;
            case uint uintValue:
                numericValue = uintValue;
                return true;
            case long longValue:
                numericValue = longValue;
                return true;
            case ulong ulongValue:
                numericValue = ulongValue;
                return true;
            case float floatValue:
                numericValue = floatValue;
                return true;
            case double doubleValue:
                numericValue = (float)doubleValue;
                return true;
            case decimal decimalValue:
                numericValue = (float)decimalValue;
                return true;
            default:
                numericValue = default;
                return false;
        }
    }

    private static string BuildMetricName(string prefix, string? nextSegment)
    {
        if (string.IsNullOrWhiteSpace(nextSegment))
            return NormalizeMetricName(prefix);

        return string.IsNullOrWhiteSpace(prefix)
            ? nextSegment
            : $"{NormalizeMetricName(prefix)}.{nextSegment}";
    }

    private static string NormalizeMetricName(string metricName)
        => metricName.Trim('.');

    private static void SaveModelState(TrainerRunContext runContext, object model, object optimizer, object scheduler,
        object gradientScaler, object trainingState, object epochSummary, int epoch)
    {
        string checkpointPath = Path.Combine(runContext.CheckpointsPath, $"model-epoch-{epoch:0000}.pt");
        PersistModelCheckpoint(checkpointPath, model, optimizer, scheduler, gradientScaler, trainingState, epochSummary);
    }

    private static void SaveBestModelStateIfNeeded(TrainerRunContext runContext, object model, object optimizer,
        object scheduler, object gradientScaler, object trainingState, object epochSummary, int epoch)
    {
        if (!ShouldSaveBestCheckpoint(trainingState, epochSummary))
            return;

        string checkpointPath = Path.Combine(runContext.CheckpointsPath, "model-best.pt");
        PersistModelCheckpoint(checkpointPath, model, optimizer, scheduler, gradientScaler, trainingState, epochSummary);
    }
}

internal sealed record TrainerRunContext(
    string RunDirectoryPath,
    string CheckpointsPath,
    string SettingsSnapshotPath,
    string MetricsPath);

internal sealed class MetricHistory
{
    public List<int> Epochs { get; set; } = [];
    public Dictionary<string, List<float>> Metrics { get; set; } = new(StringComparer.OrdinalIgnoreCase);
}
