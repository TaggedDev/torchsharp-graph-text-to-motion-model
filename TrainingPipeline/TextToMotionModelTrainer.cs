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


        var device = ResolveDevice(_settings.Device);
        _textToMotionModel = _textToMotionModel.to(device);

        int startEpoch = 1;
        if (_settings.LoadCheckpoint)
            startEpoch = checkpointService.RestoreCheckpoint(runDirectoryPath, _textToMotionModel) + 1;

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

            var trainLoss = RunEpoch(
                _textToMotionModel,
                dataset.Train,
                optimizer,
                batchSize: Math.Max(1, _settings.BatchSize),
                device: device,
                training: true,
                parentBar: epochBar);
            
            var valLoss = RunEpoch(
                _textToMotionModel,
                dataset.Val,
                optimizer: null,
                batchSize: Math.Max(1, _settings.EvaluationBatchSize),
                device: device,
                training: false,
                parentBar: epochBar);

            epochTimer.Stop();
            
            checkpointService.SaveEpochCheckpoint(checkpointsPath, _textToMotionModel, epoch);

            epochBar.Tick(
                $"Epoch {epoch}/{maxEpochs} | " +
                $"train: {trainLoss:F6} | val: {valLoss:F6} | " +
                $"{epochTimer.Elapsed.TotalSeconds:F1}s");
        }
        
        var testLoss = RunEpoch(
            _textToMotionModel,
            dataset.Test,
            optimizer: null,
            batchSize: Math.Max(1, _settings.EvaluationBatchSize),
            device: device,
            training: false);
        
        checkpointService.SaveFinalArtifacts(runDirectoryPath, testMetricsPath, _textToMotionModel);

        Console.WriteLine(
            $"Training finished. Epochs: {maxEpochs}, " +
            $"test loss: {testLoss:F6}");

        _perfMonitor.PrintSummary();
        _fidProjection?.Dispose();
    }

    private EpochResult RunEpoch(Module<Tensor, Tensor> model,
        IReadOnlyList<MotionSample> samples,
        optim.Optimizer? optimizer,
        int batchSize,
        Device device,
        bool training,
        ProgressBarBase? parentBar = null)
    {
        if (samples.Count == 0)
            return new EpochResult(0f);

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
            
            totalLoss += loss.ToSingle();
            numBatches++;
            batchBar?.Tick($"Batch {numBatches}/{totalBatches} | loss: {loss.ToSingle():F4}");
        }

        float avgLoss = numBatches > 0 ? totalLoss / numBatches : 0f;
        return new EpochResult(avgLoss);
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
        float Loss
    );

}