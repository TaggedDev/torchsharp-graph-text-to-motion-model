using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Configuration;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Training;

public sealed class Trainer
{
    private const int SaveCheckpointEach = 5;
    private const float WeightDecay = 1e-4f;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly Device _device;
    private readonly int _batchSize;
    private readonly int _maxEpochs;
    private readonly int _logEveryNSteps;
    private readonly int _numTimesteps;
    private readonly int _nodeHidden;
    private readonly int _numGcnLayers;
    private readonly int _numHeads;
    private readonly float _learningRate;
    private readonly float _gradClipNorm;
    private readonly string _checkpointDir;
    private readonly string? _resumeCheckpointPath;

    private readonly float _smoothWeight;
    private readonly float _velocityWeight;
    private readonly float _cfgDropProb;

    private readonly MotionDataset _trainSet;
    private readonly MotionDataset _valSet;
    private readonly MotionDataset _testSet;
    private readonly GraphDenoiser _denoiser;
    private readonly DdpmScheduler _scheduler;
    private readonly PositionLoss _positionLoss;
    private readonly AdamW _optimizer;

    private int _startEpoch = 1;
    private long _globalStep;

    public Trainer(IConfigurationRoot config)
    {
        var processedPath = config["ProcessedPath"] ?? "processed";
        _batchSize = int.Parse(config["BatchSize"] ?? "32");
        _maxEpochs = int.Parse(config["MaxEpochs"] ?? "100");
        _logEveryNSteps = int.Parse(config["LogEveryNSteps"] ?? "50");
        _gradClipNorm = float.Parse(config["GradClipNorm"] ?? "1.0", CultureInfo.InvariantCulture);
        _checkpointDir = config["CheckpointDir"] ?? "checkpoints";
        var deviceName = config["Device"] ?? "cuda";

        _numTimesteps = int.Parse(config["NumTimesteps"] ?? "1000");
        _nodeHidden = int.Parse(config["NodeHidden"] ?? "64");
        _numGcnLayers = int.Parse(config["NumGcnLayers"] ?? "4");
        _numHeads = int.Parse(config["NumHeads"] ?? "4");
        _learningRate = float.Parse(config["LearningRate"] ?? "0.0001", CultureInfo.InvariantCulture);
        _smoothWeight = float.Parse(config["SmoothWeight"] ?? "3.0", CultureInfo.InvariantCulture);
        _velocityWeight = float.Parse(config["VelocityWeight"] ?? "1.0", CultureInfo.InvariantCulture);
        _cfgDropProb = float.Parse(config["CfgDropProb"] ?? "0.15", CultureInfo.InvariantCulture);

        _device = new Device(deviceName);

        _trainSet = new MotionDataset(Path.Combine(processedPath, "train"), processedPath);
        _valSet = new MotionDataset(Path.Combine(processedPath, "val"), processedPath);
        _testSet = new MotionDataset(Path.Combine(processedPath, "test"), processedPath);

        Console.WriteLine($"Dataset: train={_trainSet.Count}, val={_valSet.Count}, test={_testSet.Count}");

        _scheduler = new DdpmScheduler(_numTimesteps);
        _scheduler.To(_device);

        _positionLoss = new PositionLoss(processedPath, _device);

        _denoiser = new GraphDenoiser(_numGcnLayers, _nodeHidden, _numHeads);
        _denoiser.to(_device);
        _denoiser.MoveGcnBuffers(_device);

        _optimizer = optim.AdamW(_denoiser.parameters(), lr: _learningRate, weight_decay: WeightDecay);

        Directory.CreateDirectory(_checkpointDir);

        _resumeCheckpointPath = ResolveResumeCheckpointPath(config["ResumeCheckpoint"]);
        if (!string.IsNullOrWhiteSpace(_resumeCheckpointPath))
            LoadCheckpoint(_resumeCheckpointPath);
    }

    public void Run()
    {
        if (_startEpoch > _maxEpochs)
        {
            Console.WriteLine(
                $"Checkpoint epoch {_startEpoch - 1} already meets MaxEpochs={_maxEpochs}. Skipping training loop.");
        }
        else
        {
            for (int epoch = _startEpoch; epoch <= _maxEpochs; epoch++)
            {
                var trainLoss = TrainEpoch(epoch);
                var (valLoss, _) = ComputeMetrics(_valSet);
                var (trainEvalLoss, _) = ComputeMetrics(_trainSet);

                Console.WriteLine(
                    $"Epoch {epoch}/{_maxEpochs}  " +
                    $"train_loss={trainLoss:F4}  " +
                    $"train_eval_loss={trainEvalLoss:F4}  " +
                    $"val_loss={valLoss:F4}");

                SaveLatestCheckpoint(epoch);
                if (epoch % SaveCheckpointEach == 0)
                    SaveArchiveCheckpoint(epoch);
            }
        }

        Console.WriteLine("Training complete. Running test evaluation...");
        var (testLoss, testMetrics) = ComputeMetrics(_testSet);
        Console.WriteLine($"Test loss={testLoss:F4}  metrics=[{FormatMetrics(testMetrics)}]");
    }

    private float TrainEpoch(int epoch)
    {
        _denoiser.train();
        var batches = _trainSet.GetEpochBatches(_batchSize, shuffle: true);
        double totalBatchSeconds = 0;
        int step = 0;
        int stepsSinceLog = 0;

        // Keep loss accumulation on the device and only sync at reporting points.
        using var epochLossSum = zeros(1, device: _device);
        using var runningLossSum = zeros(1, device: _device);

        foreach (var indices in batches)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            using var scope = NewDisposeScope();
            using var batch = _trainSet.LoadBatch(indices, _device);

            int b = (int)batch.Motion.shape[0];
            var t = _scheduler.SampleTimesteps(b, _device);

            Tensor noise;
            using (no_grad())
                noise = randn_like(batch.Motion) * batch.Mask.unsqueeze(-1);

            Tensor cond;
            if (_cfgDropProb > 0f)
            {
                using (no_grad())
                {
                    var dropMask = rand(b, device: _device) < _cfgDropProb;
                    var nullCond = _denoiser.NullCondition.expand(b, -1);
                    cond = where(dropMask.unsqueeze(-1), nullCond, batch.Condition);
                }
            }
            else
            {
                cond = batch.Condition;
            }

            var xt = _scheduler.QSample(batch.Motion, t, noise);
            var predicted = _denoiser.forward(xt, t, cond);
            var predictedX0 = _scheduler.PredictX0(xt, t, predicted).clamp(-5, 5);
            var (loss, _, _, _, _) = _positionLoss.Compute(
                predictedX0, batch.Motion, batch.Mask, _smoothWeight, _velocityWeight);

            if (!IsFiniteScalar(loss))
                throw new InvalidOperationException(
                    $"Non-finite training loss at epoch {epoch}, step {step + 1}. Training diverged before optimizer step.");

            _optimizer.zero_grad();
            loss.backward();
            if (_gradClipNorm > 0f)
                nn.utils.clip_grad_norm_(_denoiser.parameters(), _gradClipNorm);
            _optimizer.step();
            _globalStep++;

            using (no_grad())
            {
                var detached = loss.detach();
                epochLossSum.add_(detached);
                runningLossSum.add_(detached);
            }

            step++;
            stepsSinceLog++;

            sw.Stop();
            totalBatchSeconds += sw.Elapsed.TotalSeconds;

            if (step % _logEveryNSteps == 0)
            {
                float avgLoss = (runningLossSum / stepsSinceLog).item<float>();
                runningLossSum.zero_();
                stepsSinceLog = 0;

                double avgBatchSeconds = totalBatchSeconds / step;
                Console.WriteLine(
                    $"  Epoch {epoch} step {step}/{batches.Length} " +
                    $"avg_loss={avgLoss:F4} avg_batch_time={avgBatchSeconds:F4}s");
            }
        }

        return epochLossSum.item<float>() / Math.Max(step, 1);
    }

    private (float avgLoss, Dictionary<string, float> metrics) ComputeMetrics(
        MotionDataset dataset)
    {
        _denoiser.eval();
        var batches = dataset.GetEpochBatches(_batchSize, shuffle: false);
        int count = 0;

        using var lossSum = zeros(1, device: _device);

        using (no_grad())
        {
            foreach (var indices in batches)
            {
                using var scope = NewDisposeScope();
                using var batch = dataset.LoadBatch(indices, _device);

                int b = (int)batch.Motion.shape[0];
                var t = _scheduler.SampleTimesteps(b, _device);

                var noise = randn_like(batch.Motion) * batch.Mask.unsqueeze(-1);
                var xt = _scheduler.QSample(batch.Motion, t, noise);
                var predicted = _denoiser.forward(xt, t, batch.Condition);
                var predictedX0 = _scheduler.PredictX0(xt, t, predicted).clamp(-5, 5);
                var (loss, _, _, _, _) = _positionLoss.Compute(
                    predictedX0, batch.Motion, batch.Mask, _smoothWeight, _velocityWeight);

                if (!IsFiniteScalar(loss))
                    throw new InvalidOperationException(
                        $"Non-finite evaluation loss on dataset batch {count + 1}. The checkpoint is numerically unstable.");

                lossSum.add_(loss);
                count++;
            }
        }

        float avgLoss = lossSum.item<float>() / Math.Max(count, 1);
        var metrics = Evaluate();
        return (avgLoss, metrics);
    }

    /// <summary>
    /// Placeholder for future metrics (FID, R-precision, diversity, etc.).
    /// </summary>
    private static Dictionary<string, float> Evaluate()
    {
        return new Dictionary<string, float>();
    }

    private void SaveArchiveCheckpoint(int epoch)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        var modelPath = Path.Combine(_checkpointDir, $"model_epoch{epoch}_{timestamp}.pt");
        SaveCheckpointFiles(modelPath, epoch, "archive");
    }

    private void SaveLatestCheckpoint(int epoch)
    {
        var modelPath = Path.Combine(_checkpointDir, "latest.pt");
        SaveCheckpointFiles(modelPath, epoch, "latest");
    }

    private void SaveCheckpointFiles(string modelPath, int epoch, string label)
    {
        var optimizerPath = GetOptimizerStatePath(modelPath);
        var metadataPath = GetMetadataPath(modelPath);

        _denoiser.save(modelPath);
        _optimizer.save_state_dict(optimizerPath);

        var metadata = new CheckpointMetadata
        {
            Epoch = epoch,
            GlobalStep = _globalStep,
            NumTimesteps = _numTimesteps,
            NodeHidden = _nodeHidden,
            NumGcnLayers = _numGcnLayers,
            NumHeads = _numHeads,
            LearningRate = _learningRate,
            WeightDecay = WeightDecay,
            GradClipNorm = _gradClipNorm,
            SmoothWeight = _smoothWeight,
            VelocityWeight = _velocityWeight,
            CfgDropProb = _cfgDropProb,
            SavedAtUtc = DateTime.UtcNow
        };

        File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, JsonOptions));
        Console.WriteLine($"  {label} checkpoint saved: {modelPath}");
    }

    private void LoadCheckpoint(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Resume checkpoint was not found: {modelPath}", modelPath);

        _denoiser.load(modelPath);
        _denoiser.to(_device);
        _denoiser.MoveGcnBuffers(_device);

        var optimizerPath = GetOptimizerStatePath(modelPath);
        var metadataPath = GetMetadataPath(modelPath);
        var metadata = LoadCheckpointMetadata(metadataPath);

        if (metadata is not null)
            ValidateCheckpointMetadata(metadata, modelPath);

        if (File.Exists(optimizerPath))
        {
            _optimizer.load_state_dict(optimizerPath);
            Console.WriteLine($"Loaded optimizer state: {optimizerPath}");
        }
        else
        {
            Console.WriteLine(
                $"Optimizer state not found for '{modelPath}'. Resuming with model weights only and a fresh optimizer.");
        }

        var completedEpoch = metadata?.Epoch ?? TryParseEpochFromPath(modelPath) ?? 0;
        _globalStep = metadata?.GlobalStep ?? 0;
        _startEpoch = Math.Max(completedEpoch + 1, 1);

        if (metadata is null)
        {
            Console.WriteLine(
                $"Checkpoint metadata not found for '{modelPath}'. " +
                $"Using filename-derived epoch={completedEpoch}.");
        }

        Console.WriteLine(
            $"Resumed from '{modelPath}'. Completed epoch={completedEpoch}, next epoch={_startEpoch}, global_step={_globalStep}.");
    }

    private CheckpointMetadata? LoadCheckpointMetadata(string metadataPath)
    {
        if (!File.Exists(metadataPath))
            return null;

        var json = File.ReadAllText(metadataPath);
        var metadata = JsonSerializer.Deserialize<CheckpointMetadata>(json);
        if (metadata is null)
            throw new InvalidOperationException($"Checkpoint metadata file is empty: {metadataPath}");
        return metadata;
    }

    private void ValidateCheckpointMetadata(CheckpointMetadata metadata, string modelPath)
    {
        if (metadata.NumTimesteps != _numTimesteps)
            throw new InvalidOperationException(
                $"Checkpoint '{modelPath}' was trained with NumTimesteps={metadata.NumTimesteps}, " +
                $"but current config expects {_numTimesteps}.");
        if (metadata.NodeHidden != _nodeHidden)
            throw new InvalidOperationException(
                $"Checkpoint '{modelPath}' was trained with NodeHidden={metadata.NodeHidden}, " +
                $"but current config expects {_nodeHidden}.");
        if (metadata.NumGcnLayers != _numGcnLayers)
            throw new InvalidOperationException(
                $"Checkpoint '{modelPath}' was trained with NumGcnLayers={metadata.NumGcnLayers}, " +
                $"but current config expects {_numGcnLayers}.");
        if (metadata.NumHeads != _numHeads)
            throw new InvalidOperationException(
                $"Checkpoint '{modelPath}' was trained with NumHeads={metadata.NumHeads}, " +
                $"but current config expects {_numHeads}.");
    }

    private string? ResolveResumeCheckpointPath(string? checkpointPath)
    {
        if (string.IsNullOrWhiteSpace(checkpointPath))
            return null;

        var trimmed = checkpointPath.Trim();
        if (Path.IsPathRooted(trimmed))
            return Path.GetFullPath(trimmed);

        var fromWorkingDirectory = Path.GetFullPath(trimmed);
        if (File.Exists(fromWorkingDirectory))
            return fromWorkingDirectory;

        return Path.GetFullPath(Path.Combine(_checkpointDir, trimmed));
    }

    private static int? TryParseEpochFromPath(string modelPath)
    {
        var fileName = Path.GetFileNameWithoutExtension(modelPath);
        var match = Regex.Match(fileName, @"(?:^|_)epoch(?<epoch>\d+)(?:_|$)", RegexOptions.IgnoreCase);
        if (!match.Success)
            return null;

        if (!int.TryParse(match.Groups["epoch"].Value, NumberStyles.None, CultureInfo.InvariantCulture, out var epoch))
            return null;

        return epoch;
    }

    private static string GetOptimizerStatePath(string modelPath)
    {
        var directory = Path.GetDirectoryName(modelPath) ?? ".";
        var fileName = Path.GetFileNameWithoutExtension(modelPath);
        return Path.Combine(directory, $"{fileName}.optimizer.pt");
    }

    private static string GetMetadataPath(string modelPath)
    {
        var directory = Path.GetDirectoryName(modelPath) ?? ".";
        var fileName = Path.GetFileNameWithoutExtension(modelPath);
        return Path.Combine(directory, $"{fileName}.metadata.json");
    }

    private static string FormatMetrics(Dictionary<string, float> metrics)
    {
        if (metrics.Count == 0)
            return "no metrics yet";
        return string.Join(", ", metrics.Select(kv => $"{kv.Key}={kv.Value:F4}"));
    }

    private static bool IsFiniteScalar(Tensor value)
    {
        var scalar = value.item<float>();
        return !float.IsNaN(scalar) && !float.IsInfinity(scalar);
    }

    private sealed class CheckpointMetadata
    {
        public int Epoch { get; set; }
        public long GlobalStep { get; set; }
        public int NumTimesteps { get; set; }
        public int NodeHidden { get; set; }
        public int NumGcnLayers { get; set; }
        public int NumHeads { get; set; }
        public float LearningRate { get; set; }
        public float WeightDecay { get; set; }
        public float GradClipNorm { get; set; }
        public float SmoothWeight { get; set; }
        public float VelocityWeight { get; set; }
        public float CfgDropProb { get; set; }
        public DateTime SavedAtUtc { get; set; }
    }
}
