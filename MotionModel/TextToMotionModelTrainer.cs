using System.Diagnostics;
using Microsoft.Extensions.Options;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class TextToMotionModelTrainer(
    IOptions<ModelSettings> modelOptions,
    IOptions<TrainingSettings> trainingOptions,
    ModelCheckpointService checkpointService,
    TrainingMetricsService metricsService)
{
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

        metricsService.Initialize(metricsPath, _trainingSettings.LoadCheckpoint);
        var model = new StubTextToMotionModel();
        var optimizer = optim.Adam(
            model.parameters(),
            lr: _modelSettings.LearningRate,
            weight_decay: _modelSettings.WeightDecay);

        int startEpoch = 1;
        if (_trainingSettings.LoadCheckpoint)
            startEpoch = checkpointService.RestoreCheckpoint(runDirectoryPath, model, metricsService.Log) + 1;

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

            var valSnapshot = EvaluateMotionMetrics(
                model,
                Math.Max(1, _modelSettings.EvaluationBatchSize),
                epoch,
                MotionEvalPhase.Validation);

            epochTimer.Stop();

            metricsService.RecordEpoch(
                epoch,
                trainingMetrics.Loss,
                trainingMetrics.Metric,
                validationMetrics.Loss,
                validationMetrics.Metric,
                (float)epochTimer.Elapsed.TotalSeconds);

            metricsService.RecordMotionMetrics(valSnapshot);

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

        var testSnapshot = EvaluateMotionMetrics(
            model,
            Math.Max(1, _modelSettings.EvaluationBatchSize),
            metricsService.Log.Epochs.LastOrDefault(),
            MotionEvalPhase.Test);

        var testMetricsLog = metricsService.CreateTestMetricsLog(
            metricsService.Log.Epochs.LastOrDefault(),
            testingMetrics.Loss,
            testingMetrics.Metric);

        metricsService.RecordMotionMetrics(testSnapshot);
        checkpointService.SaveFinalArtifacts(runDirectoryPath, testMetricsPath, model, testMetricsLog);

        Console.WriteLine(
            $"Training finished. Epochs: {metricsService.Log.Epochs.LastOrDefault()}, " +
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

    private static MotionEvalSnapshot EvaluateMotionMetrics(
        StubTextToMotionModel model,
        int batchSize,
        int epoch,
        MotionEvalPhase phase)
    {
        using var scope = NewDisposeScope();

        var motionFeatures = randn([batchSize, StubTextToMotionModel.OutputFeatures]);
        var textFeatures = randn([batchSize, StubTextToMotionModel.OutputFeatures]);

        return new MotionEvalSnapshot(
            epoch, phase,
            RPrecisionMetric.Compute(motionFeatures, textFeatures, topK: 1),
            RPrecisionMetric.Compute(motionFeatures, textFeatures, topK: 2),
            RPrecisionMetric.Compute(motionFeatures, textFeatures, topK: 3),
            FrechetInceptionDistanceMetric.Compute(motionFeatures, textFeatures),
            MultimodalDistanceMetric.Compute(motionFeatures, textFeatures),
            DiversityMetric.Compute(motionFeatures),
            MultimodalityMetric.Compute(motionFeatures, numModalities: 10));
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

    private static void SetRandomSeed(int seed)
    {
        long normalizedSeed = Math.Max(0, seed);
        random.manual_seed(normalizedSeed);
        if (cuda.is_available())
            cuda.manual_seed(normalizedSeed);
    }

    private sealed record StubEpochMetrics(float Loss, float Metric);
}
