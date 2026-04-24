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
    Module<Tensor, Tensor> textToMotionModel,
    HumanML3DDataset dataset)
{
    private readonly TrainingSettings _settings = trainingOptions.Value;
    private readonly PerformanceMonitor _perfMonitor = new();
    private Module<Tensor, Tensor> _textToMotionModel = textToMotionModel;
    private Tensor? _fidProjection;

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
        _textToMotionModel = _textToMotionModel.to(device);

        int startEpoch = 1;
        if (_settings.LoadCheckpoint)
            startEpoch = checkpointService.RestoreCheckpoint(runDirectoryPath, _textToMotionModel, metricsService.Log) + 1;

        if (startEpoch > maxEpochs)
        {
            Console.WriteLine(
                $"No training steps executed. Resume epoch {startEpoch} is greater than configured max epoch {maxEpochs}.");
            return;
        }

        var optimizer = optim.AdamW(
            _textToMotionModel.parameters(),
            lr: _settings.LearningRate,
            weight_decay: _settings.WeightDecay);

        int outputDim = 15780;
        _fidProjection = torch.randn(outputDim, 512) / MathF.Sqrt(512);

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

            var (trainLoss, _) = RunEpoch(
                _textToMotionModel,
                dataset.Train,
                optimizer,
                batchSize: Math.Max(1, _settings.BatchSize),
                device,
                training: true,
                computeMetrics: false,
                epoch: epoch,
                parentBar: epochBar);
            
            var (valLoss, valSnapshot) = RunEpoch(
                _textToMotionModel,
                dataset.Val,
                optimizer: null,
                batchSize: Math.Max(1, _settings.EvaluationBatchSize),
                device,
                training: false,
                computeMetrics: true,
                epoch: epoch,
                parentBar: epochBar);

            epochTimer.Stop();

            metricsService.RecordEpoch(
                epoch,
                trainLoss,
                valLoss,
                (float)epochTimer.Elapsed.TotalSeconds);

            if (valSnapshot == null)
                continue;
            
            metricsService.RecordMotionMetrics(valSnapshot);
            checkpointService.SaveEpochCheckpoint(checkpointsPath, _textToMotionModel, epoch);

            epochBar.Tick(
                $"Epoch {epoch}/{maxEpochs} | " +
                $"train: {trainLoss:F6} | val: {valLoss:F6} | " +
                $"FID: {valSnapshot.FrechetInceptionDistance:F4} | " +
                $"MM: {valSnapshot.Multimodality:F4} | " +
                $"MM Dist: {valSnapshot.MultimodalDistance:F4} | " +
                $"R@1: {valSnapshot.RPrecisionTop1:F4} | " +
                $"R@2: {valSnapshot.RPrecisionTop2:F4} | " +
                $"R@3: {valSnapshot.RPrecisionTop3:F4} | " +
                $"Div: {valSnapshot.Diversity:F4} | " +
                $"{epochTimer.Elapsed.TotalSeconds:F1}s");
        }
        
        var (testLoss, testSnapshot) = RunEpoch(
            _textToMotionModel,
            dataset.Test,
            optimizer: null,
            batchSize: Math.Max(1, _settings.EvaluationBatchSize),
            device,
            training: false,
            computeMetrics: true,
            epoch: maxEpochs);

        var testMetricsLog = metricsService.CreateTestMetricsLog(
            metricsService.Log.Epochs.LastOrDefault(),
            testLoss);

        if (testSnapshot != null)
            metricsService.RecordMotionMetrics(testSnapshot);
        
        checkpointService.SaveFinalArtifacts(runDirectoryPath, testMetricsPath, _textToMotionModel, testMetricsLog);

        Console.WriteLine(
            $"Training finished. Epochs: {metricsService.Log.Epochs.LastOrDefault()}, " +
            $"test loss: {testLoss:F6}");

        _perfMonitor.PrintSummary();
        _fidProjection?.Dispose();
    }

    private EpochResult RunEpoch(
        Module<Tensor, Tensor> model,
        IReadOnlyList<MotionSample> samples,
        optim.Optimizer? optimizer,
        int batchSize,
        Device device,
        bool training,
        bool computeMetrics = false,
        int epoch = 0,
        ProgressBarBase? parentBar = null)
    {
        if (samples.Count == 0)
            return new EpochResult(0f, null);

        float totalLoss = 0f;
        int numBatches = 0;

        var indices = Enumerable.Range(0, samples.Count).ToList();
        if (training)
            indices = indices.OrderBy(_ => Random.Shared.Next()).ToList();

        int totalBatches = (int)Math.Ceiling((double)indices.Count / batchSize);
        string phaseName = training ? "Train" : "Val/Test";

        var childOptions = new ProgressBarOptions
        {
            ForegroundColor = ConsoleColor.Yellow,
            BackgroundColor = ConsoleColor.DarkGray,
            ProgressCharacter = '─',
            DisplayTimeInRealTime = false,
        };

        using var batchBar = parentBar?.Spawn(totalBatches, phaseName, childOptions);

        model.train(training);
        using var noGradGuard = training ? null : torch.no_grad();

        // metric accumulators
        WelfordCovarianceAccumulator? realAcc = computeMetrics ? new() : null;
        WelfordCovarianceAccumulator? genAcc = computeMetrics ? new() : null;
        var diversityReservoir = computeMetrics ? new List<Tensor>() : null;
        int diversitySeen = 0;

        for (int i = 0; i < indices.Count; i += batchSize)
        {
            var batchIndices = indices.Skip(i).Take(batchSize).ToList();

            var (textEmb, motionFrames) = dataset.GetBatch(samples, batchIndices, device);

            var predicted = model.forward(textEmb);

            var loss = functional.mse_loss(predicted, motionFrames);

            if (training && optimizer is not null)
            {
                using var scope = NewDisposeScope();
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }

            if (computeMetrics)
            {
                using var proj = _fidProjection!.to(device);
                using var projG = predicted.detach().mm(proj);
                genAcc!.Update(projG);

                using var projR = motionFrames.detach().mm(proj);
                realAcc!.Update(projR);

                int B = (int)predicted.shape[0];
                for (int s = 0; s < B; s++)
                {
                    diversitySeen++;
                    if (diversityReservoir!.Count < 300)
                        diversityReservoir.Add(predicted[s].detach().cpu());
                    else
                    {
                        int j = Random.Shared.Next(diversitySeen);
                        if (j < 300)
                            diversityReservoir[j] = predicted[s].detach().cpu();
                    }
                }
            }

            totalLoss += loss.ToSingle();
            numBatches++;
            batchBar?.Tick($"Batch {numBatches}/{totalBatches} | loss: {loss.ToSingle():F4}");
        }

        float avgLoss = numBatches > 0 ? totalLoss / numBatches : 0f;

        if (!computeMetrics)
            return new EpochResult(avgLoss, null);

        using var metricScope = NewDisposeScope();

        var (muR, sigmaR) = realAcc!.Finalize();
        var (muG, sigmaG) = genAcc!.Finalize();

        float frechetInceptionDistance = FrechetInceptionDistanceMetric.Compute(muR, sigmaR, muG, sigmaG);

        var diversityTensor = torch.stack(diversityReservoir!.ToArray(), dim: 0);
        float diversity = DiversityMetric.Compute(diversityTensor);

        float multimodality = MultimodalityMetric.Compute(null, numModalities: 2);
        float rPrecisionTop1 = RPrecisionMetric.Compute(null, null, topK: 1);
        float rPrecisionTop2 = RPrecisionMetric.Compute(null, null, topK: 2);
        float rPrecisionTop3 = RPrecisionMetric.Compute(null, null, topK: 3);
        float multimodalDistance = MultimodalDistanceMetric.Compute(null, null);

        var snapshot = new MotionEvalSnapshot(
            epoch,
            frechetInceptionDistance,
            diversity,
            multimodality,
            rPrecisionTop1,
            rPrecisionTop2,
            rPrecisionTop3,
            multimodalDistance);

        realAcc?.Dispose();
        genAcc?.Dispose();
        foreach (var t in diversityReservoir ?? new List<Tensor>())
            t.Dispose();

        return new EpochResult(avgLoss, snapshot);
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
                throw new InvalidOperationException(
                    "Training.LoadRunNumber must be greater than 0 when Training.LoadCheckpoint is true.");

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

                string[] parts = name.Split('-',
                    StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
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
        return new Device(DeviceType.CPU);
    }

    public sealed record EpochResult(
        float Loss,
        MotionEvalSnapshot? Metrics
    );

}