using System.Diagnostics;
using Microsoft.Extensions.Options;
using Text2Motion.Dataset;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class TextToMotionModelTrainer(
    IOptions<ModelSettings> modelOptions,
    IOptions<TrainingSettings> trainingOptions,
    ModelCheckpointService checkpointService,
    TrainingMetricsService metricsService,
    Module<Tensor, Tensor> model,
    HumanML3DDataset dataset)
{
    private readonly ModelSettings _modelSettings = modelOptions.Value;
    private readonly TrainingSettings _trainingSettings = trainingOptions.Value;

    public async Task TrainAsync(CancellationToken token)
    {
        int maxEpochs = Math.Max(1, _modelSettings.MaxEpochs);
        int printEveryEpoch = Math.Max(1, _modelSettings.PrintEveryEpoch);

        SetRandomSeed(_modelSettings.RandomSeed);
        await dataset.LoadAsync();

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
        var optimizer = optim.AdamW(
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
            return;
        }

        var device = ResolveDevice(_modelSettings.Device);
        model = model.to(device);

        for (int epoch = startEpoch; epoch <= maxEpochs; epoch++)
        {
            token.ThrowIfCancellationRequested();

            var epochTimer = Stopwatch.StartNew();
            Console.WriteLine($"Epoch {epoch}/{maxEpochs} started.");

            model.train();
            var trainingMetrics = RunEpoch(
                model,
                dataset.Train,
                optimizer,
                batchSize: Math.Max(1, _modelSettings.BatchSize),
                device,
                training: true);

            model.eval();
            var validationMetrics = RunEpoch(
                model,
                dataset.Val,
                optimizer: null,
                batchSize: Math.Max(1, _modelSettings.EvaluationBatchSize),
                device,
                training: false);

            var valSnapshot = EvaluateMotionMetrics(
                model,
                dataset.Val,
                device,
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

        var testingMetrics = RunEpoch(
            model,
            dataset.Test,
            optimizer: null,
            batchSize: Math.Max(1, _modelSettings.EvaluationBatchSize),
            device,
            training: false);

        var testSnapshot = EvaluateMotionMetrics(
            model,
            dataset.Test,
            device,
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
    }

    private StubEpochMetrics RunEpoch(
        Module<Tensor, Tensor> model,
        IReadOnlyList<MotionSample> samples,
        optim.Optimizer? optimizer,
        int batchSize,
        Device device,
        bool training)
    {
        if (samples.Count == 0)
            return new StubEpochMetrics(0f, 0f);

        float totalLoss = 0f;
        int numBatches = 0;

        var indices = Enumerable.Range(0, samples.Count).ToList();
        if (training)
            indices = indices.OrderBy(_ => Random.Shared.Next()).ToList();

        for (int i = 0; i < indices.Count; i += batchSize)
        {
            using var scope = NewDisposeScope();

            var batchIndices = indices.Skip(i).Take(batchSize).ToList();
            var (textEmb, motionFrames) = dataset.GetBatch(samples, batchIndices, device);

            var predicted = model.forward(textEmb);
            var loss = functional.mse_loss(predicted, motionFrames);

            if (training && optimizer is not null)
            {
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }

            totalLoss += loss.ToSingle();
            numBatches++;
        }

        float avgLoss = numBatches > 0 ? totalLoss / numBatches : 0f;
        return new StubEpochMetrics(avgLoss, 1.0f / (1.0f + avgLoss));
    }

    private MotionEvalSnapshot EvaluateMotionMetrics(
        Module<Tensor, Tensor> model,
        IReadOnlyList<MotionSample> samples,
        Device device,
        int batchSize,
        int epoch,
        MotionEvalPhase phase)
    {
        using var scope = NewDisposeScope();

        if (samples.Count == 0)
        {
            return new MotionEvalSnapshot(epoch, phase, 0f);
        }

        int sampleIdx = Random.Shared.Next(Math.Min(batchSize, samples.Count));
        var evalIndices = Enumerable.Range(sampleIdx, Math.Min(batchSize, samples.Count - sampleIdx)).ToList();
        var (textEmb, motionFrames) = dataset.GetBatch(samples, evalIndices, device);

        var predicted = model.forward(textEmb);
        float l2Distance = L2DistanceMetric.Compute(predicted, motionFrames);

        return new MotionEvalSnapshot(epoch, phase, l2Distance);
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

    private static Device ResolveDevice(string deviceStr)
    {
        if (deviceStr.Equals("cuda", StringComparison.OrdinalIgnoreCase) && cuda.is_available())
            return new Device(DeviceType.CUDA, 0);
        return new Device(DeviceType.CPU, -1);
    }

    private sealed record StubEpochMetrics(float Loss, float Metric);
}
