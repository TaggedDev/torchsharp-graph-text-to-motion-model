using System.Diagnostics;
using Microsoft.Extensions.Options;
using ShellProgressBar;
using Text2Motion.Dataset;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class TextToMotionModelTrainer(
    IOptions<TrainingSettings> trainingOptions,
    ModelCheckpointService checkpointService,
    TrainingMetricsService metricsService,
    Module<Tensor, Tensor> model,
    HumanML3DDataset dataset)
{
    private readonly TrainingSettings _settings = trainingOptions.Value;
    private readonly PerformanceMonitor _perfMonitor = new();

    public async Task TrainAsync(CancellationToken token)
    {
        int maxEpochs = Math.Max(1, _settings.MaxEpochs);

        SetRandomSeed(_settings.RandomSeed);
        await dataset.LoadAsync();

        string outputRootPath = ResolveOutputRootPath(_settings);
        string runDirectoryPath = ResolveRunDirectory(outputRootPath, _settings);
        string checkpointsPath = Path.Combine(runDirectoryPath, "checkpoints");
        string resultsPath = Path.Combine(runDirectoryPath, "results");
        string metricsPath = Path.Combine(resultsPath, "metrics.json");
        string testMetricsPath = Path.Combine(resultsPath, "test-metrics.json");

        Directory.CreateDirectory(runDirectoryPath);
        Directory.CreateDirectory(checkpointsPath);
        Directory.CreateDirectory(resultsPath);

        metricsService.Initialize(metricsPath, _settings.LoadCheckpoint);

        var device = ResolveDevice(_settings.Device);
        model = model.to(device);

        int startEpoch = 1;
        if (_settings.LoadCheckpoint)
            startEpoch = checkpointService.RestoreCheckpoint(runDirectoryPath, model, metricsService.Log) + 1;

        if (startEpoch > maxEpochs)
        {
            Console.WriteLine(
                $"No training steps executed. Resume epoch {startEpoch} is greater than configured max epoch {maxEpochs}.");
            return;
        }

        var optimizer = optim.AdamW(
            model.parameters(),
            lr: _settings.LearningRate,
            weight_decay: _settings.WeightDecay);

        CudaDebugger.PrintDeviceInfo(device);

        int totalEpochs = maxEpochs - startEpoch + 1;
        var epochBarOptions = new ProgressBarOptions
        {
            ForegroundColor = ConsoleColor.Cyan,
            BackgroundColor = ConsoleColor.DarkGray,
            ProgressCharacter = '─',
            DisplayTimeInRealTime = false,
        };
        using var epochBar = new ProgressBar(totalEpochs, "Training epochs", epochBarOptions);

        for (int epoch = startEpoch; epoch <= maxEpochs; epoch++)
        {
            token.ThrowIfCancellationRequested();

            var epochTimer = Stopwatch.StartNew();

            model.train();
            float trainLoss = RunEpoch(
                model,
                dataset.Train,
                optimizer,
                batchSize: Math.Max(1, _settings.BatchSize),
                device,
                training: true,
                parentBar: epochBar);

            model.eval();
            float valLoss = RunEpoch(
                model,
                dataset.Val,
                optimizer: null,
                batchSize: Math.Max(1, _settings.EvaluationBatchSize),
                device,
                training: false,
                parentBar: epochBar);

            var valSnapshot = EvaluateMotionMetrics(
                model,
                dataset.Val,
                device,
                Math.Max(1, _settings.EvaluationBatchSize),
                epoch,
                MotionEvalPhase.Validation);

            epochTimer.Stop();

            metricsService.RecordEpoch(
                epoch,
                trainLoss,
                valLoss,
                (float)epochTimer.Elapsed.TotalSeconds);

            metricsService.RecordMotionMetrics(valSnapshot);

            checkpointService.SaveEpochCheckpoint(checkpointsPath, model, epoch);

            epochBar.Tick(
                $"Epoch {epoch}/{maxEpochs} | " +
                $"train: {trainLoss:F6} | val: {valLoss:F6} | " +
                $"FID: {valSnapshot.FrechetInceptionDistance:F4} | " +
                $"R@1: {valSnapshot.Multimodality:F4} | " +
                $"R@1: {valSnapshot.MultimodalDistance:F4} | " +
                $"R@1: {valSnapshot.RPrecisionTop1:F4} | " +
                $"R@2: {valSnapshot.RPrecisionTop2:F4} | " +
                $"R@3: {valSnapshot.RPrecisionTop3:F4} | " +
                $"Div: {valSnapshot.Diversity:F4} | " +
                $"{epochTimer.Elapsed.TotalSeconds:F1}s");
        }

        float testLoss = RunEpoch(
            model,
            dataset.Test,
            optimizer: null,
            batchSize: Math.Max(1, _settings.EvaluationBatchSize),
            device,
            training: false);

        var testSnapshot = EvaluateMotionMetrics(
            model,
            dataset.Test,
            device,
            Math.Max(1, _settings.EvaluationBatchSize),
            metricsService.Log.Epochs.LastOrDefault(),
            MotionEvalPhase.Test);

        var testMetricsLog = metricsService.CreateTestMetricsLog(
            metricsService.Log.Epochs.LastOrDefault(),
            testLoss);

        metricsService.RecordMotionMetrics(testSnapshot);
        checkpointService.SaveFinalArtifacts(runDirectoryPath, testMetricsPath, model, testMetricsLog);

        Console.WriteLine(
            $"Training finished. Epochs: {metricsService.Log.Epochs.LastOrDefault()}, " +
            $"test loss: {testLoss:F6}");

        _perfMonitor.PrintSummary();
    }

    private float RunEpoch(
        Module<Tensor, Tensor> model,
        IReadOnlyList<MotionSample> samples,
        optim.Optimizer? optimizer,
        int batchSize,
        Device device,
        bool training,
        ProgressBarBase? parentBar = null)
    {
        if (samples.Count == 0)
            return 0f;

        float totalLoss = 0f;
        int numBatches = 0;

        var indices = Enumerable.Range(0, samples.Count).ToList();
        if (training)
            indices = indices.OrderBy(_ => Random.Shared.Next()).ToList();

        int totalBatches = (int)Math.Ceiling((double)indices.Count / batchSize);
        string phase = training ? "Train" : "Val/Test";
        var childOptions = new ProgressBarOptions
        {
            ForegroundColor = ConsoleColor.Yellow,
            BackgroundColor = ConsoleColor.DarkGray,
            ProgressCharacter = '─',
            DisplayTimeInRealTime = false,
        };
        using var batchBar = parentBar?.Spawn(totalBatches, phase, childOptions);

        for (int i = 0; i < indices.Count; i += batchSize)
        {
            using var scope = NewDisposeScope();

            var batchIndices = indices.Skip(i).Take(batchSize).ToList();

            // Profile: data loading time
            _perfMonitor.StartTimer("data_load");
            var (textEmb, motionFrames) = dataset.GetBatch(samples, batchIndices, device);
            var dataLoadMs = _perfMonitor.EndTimer("data_load");

            // Profile: forward pass
            _perfMonitor.StartTimer("forward_pass");
            var predicted = model.forward(textEmb);
            CudaDebugger.SynchronizeGpu();
            var forwardMs = _perfMonitor.EndTimer("forward_pass");

            // Profile: loss computation
            _perfMonitor.StartTimer("loss_compute");
            var loss = functional.mse_loss(predicted, motionFrames);
            var lossMs = _perfMonitor.EndTimer("loss_compute");

            if (training && optimizer is not null)
            {
                // Profile: backward pass
                _perfMonitor.StartTimer("backward_pass");
                optimizer.zero_grad();
                loss.backward();
                CudaDebugger.SynchronizeGpu();
                var backwardMs = _perfMonitor.EndTimer("backward_pass");

                // Profile: optimizer step
                _perfMonitor.StartTimer("optimizer_step");
                optimizer.step();
                CudaDebugger.SynchronizeGpu();
                var stepMs = _perfMonitor.EndTimer("optimizer_step");

                if (numBatches == 0)
                {
                    Console.WriteLine($"[TIMING] Batch 1: data={dataLoadMs}ms, fwd={forwardMs}ms, loss={lossMs}ms, bwd={backwardMs}ms, step={stepMs}ms");
                }
            }

            totalLoss += loss.ToSingle();
            numBatches++;
            batchBar?.Tick($"Batch {numBatches}/{totalBatches} | loss: {loss.ToSingle():F4}");
        }

        float avgLoss = numBatches > 0 ? totalLoss / numBatches : 0f;
        return avgLoss;
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
            return new MotionEvalSnapshot(
                epoch,
                0f,
                0f,
                0f,
                0f,
                0f,
                0f,
                0f);
        }

        model.eval();
        using var noGradGuard = torch.no_grad();

        Tensor? allPredicted = null;
        Tensor? allGroundTruth = null;
        Tensor? allTextEmb = null;

        var indices = Enumerable.Range(0, samples.Count).ToList();

        for (int i = 0; i < indices.Count; i += batchSize)
        {
            var batchIndices = indices.Skip(i).Take(batchSize).ToList();
            var (textEmb, motionFrames) = dataset.GetBatch(samples, batchIndices, device);

            var predicted = model.forward(textEmb);

            allPredicted = allPredicted is null
                ? predicted.clone()
                : torch.cat([allPredicted, predicted], dim: 0);

            allGroundTruth = allGroundTruth is null
                ? motionFrames.clone()
                : torch.cat([allGroundTruth, motionFrames], dim: 0);

            allTextEmb = allTextEmb is null
                ? textEmb.clone()
                : torch.cat([allTextEmb, textEmb], dim: 0);
        }

        float l2Distance = L2DistanceMetric.Compute(allPredicted!, allGroundTruth!);
        float frechetInceptionDistance = FrechetInceptionDistanceMetric.Compute(allGroundTruth!, allPredicted!);
        float diversity = DiversityMetric.Compute(allPredicted!);
        float multimodality = MultimodalityMetric.Compute(allPredicted!, numModalities: 2);
        float rPrecisionTop1 = RPrecisionMetric.Compute(allPredicted!, allTextEmb!, topK: 1);
        float rPrecisionTop2 = RPrecisionMetric.Compute(allPredicted!, allTextEmb!, topK: 2);
        float rPrecisionTop3 = RPrecisionMetric.Compute(allPredicted!, allTextEmb!, topK: 3);
        float multimodalDistance = MultimodalDistanceMetric.Compute(allPredicted!, allTextEmb!);

        return new MotionEvalSnapshot(
            epoch,
            frechetInceptionDistance,
            diversity,
            multimodality,
            rPrecisionTop1,
            rPrecisionTop2,
            rPrecisionTop3,
            multimodalDistance);
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

}
