using System.Globalization;
using Microsoft.Extensions.Configuration;
using TorchSharp;
using static TorchSharp.torch;

namespace Training;

public sealed class Trainer
{
    private readonly Device _device;
    private readonly int _batchSize;
    private readonly int _maxEpochs;
    private readonly int _logEveryNSteps;
    private readonly float _gradClipNorm;
    private readonly string _checkpointDir;
    private readonly string _checkpointFilePrefix;
    private readonly float _condDropoutProb;
    private readonly int _maxSequenceLength;

    private readonly MotionDataset _trainSet;
    private readonly MotionDataset _valSet;
    private readonly MotionDataset _testSet;
    private readonly GraphDenoiser _denoiser;
    private readonly DdpmScheduler _scheduler;
    private readonly optim.Optimizer _optimizer;

    public Trainer(IConfigurationRoot config)
    {
        var processedPath = config["ProcessedPath"] ?? "processed";
        _batchSize = int.Parse(config["BatchSize"] ?? "32");
        _maxEpochs = int.Parse(config["MaxEpochs"] ?? "100");
        _logEveryNSteps = int.Parse(config["LogEveryNSteps"] ?? "50");
        _gradClipNorm = float.Parse(config["GradClipNorm"] ?? "1.0", CultureInfo.InvariantCulture);

        // v2 checkpoints land in a subdirectory so legacy .pt files stay
        // accessible for A/B comparison in the Visualize UI.
        var checkpointRoot = config["CheckpointDir"] ?? "checkpoints";
        var checkpointSubdir = config["CheckpointSubdir"] ?? "v2";
        _checkpointDir = Path.Combine(checkpointRoot, checkpointSubdir);
        _checkpointFilePrefix = string.IsNullOrEmpty(checkpointSubdir) ? "model" : $"model_{checkpointSubdir}";
        _condDropoutProb = float.Parse(config["CondDropoutProb"] ?? "0.1", CultureInfo.InvariantCulture);
        _maxSequenceLength = int.Parse(config["MaxSequenceLength"] ?? "196");

        var deviceName = config["Device"] ?? "cuda";

        int numTimesteps = int.Parse(config["NumTimesteps"] ?? "1000");
        int nodeHidden = int.Parse(config["NodeHidden"] ?? "64");
        int numStBlocks = int.Parse(config["NumStBlocks"] ?? config["NumGcnLayers"] ?? "4");
        int numHeads = int.Parse(config["NumHeads"] ?? "4");
        float lr = float.Parse(config["LearningRate"] ?? "0.0001", CultureInfo.InvariantCulture);

        _device = new Device(deviceName);

        _trainSet = new MotionDataset(Path.Combine(processedPath, "train"), processedPath);
        _valSet = new MotionDataset(Path.Combine(processedPath, "val"), processedPath);
        _testSet = new MotionDataset(Path.Combine(processedPath, "test"), processedPath);

        Console.WriteLine($"Dataset: train={_trainSet.Count}, val={_valSet.Count}, test={_testSet.Count}");

        _scheduler = new DdpmScheduler(numTimesteps);
        _scheduler.To(_device);

        _denoiser = new GraphDenoiser(numStBlocks, nodeHidden, numHeads);
        _denoiser.to(_device);
        _denoiser.MoveGcnBuffers(_device);

        _optimizer = optim.AdamW(_denoiser.parameters(), lr: lr, weight_decay: 1e-4);

        Directory.CreateDirectory(_checkpointDir);
    }

    public void Run()
    {
        const int saveCheckpointEach = 5;

        for (int epoch = 1; epoch <= _maxEpochs; epoch++)
        {
            var trainLoss = TrainEpoch(epoch);
            var (valLoss, _) = ComputeMetrics(_valSet);
            var (trainEvalLoss, _) = ComputeMetrics(_trainSet);

            Console.WriteLine(
                $"Epoch {epoch}/{_maxEpochs}  " +
                $"train_loss={trainLoss:F4}  " +
                $"train_eval_loss={trainEvalLoss:F4}  " +
                $"val_loss={valLoss:F4}");

            if (epoch % saveCheckpointEach == 0)
                SaveCheckpoint(epoch);
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

        // GPU-resident loss accumulators — avoid syncing every step
        using var epochLossSum = zeros(1, device: _device);
        using var runningLossSum = zeros(1, device: _device);

        foreach (var indices in batches)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            using var scope = NewDisposeScope();
            using var batch = _trainSet.LoadBatch(indices, _device, _condDropoutProb, _maxSequenceLength);

            int b = (int)batch.Motion.shape[0];
            var t = _scheduler.SampleTimesteps(b, _device);

            Tensor noise;
            using (no_grad())
                noise = randn_like(batch.Motion) * batch.Mask.unsqueeze(-1);

            var xt = _scheduler.QSample(batch.Motion, t, noise);
            var predicted = _denoiser.forward(xt, t, batch.Condition);
            var loss = _scheduler.Loss(predicted, noise, batch.Mask);

            _optimizer.zero_grad();
            loss.backward();
            if (_gradClipNorm > 0f)
                nn.utils.clip_grad_norm_(_denoiser.parameters(), _gradClipNorm);
            _optimizer.step();

            // Accumulate on GPU; sync only at log intervals / end of epoch
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
                using var batch = dataset.LoadBatch(indices, _device, 0f, _maxSequenceLength);

                int b = (int)batch.Motion.shape[0];
                var t = _scheduler.SampleTimesteps(b, _device);

                var noise = randn_like(batch.Motion) * batch.Mask.unsqueeze(-1);
                var xt = _scheduler.QSample(batch.Motion, t, noise);
                var predicted = _denoiser.forward(xt, t, batch.Condition);
                var loss = _scheduler.Loss(predicted, noise, batch.Mask);

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
        // TODO: implement FID, R-precision, diversity, multimodality metrics
        return new Dictionary<string, float>();
    }

    private void SaveCheckpoint(int epoch)
    {
        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var path = Path.Combine(_checkpointDir, $"{_checkpointFilePrefix}_epoch{epoch}_{timestamp}.pt");
        _denoiser.save(path);
        Console.WriteLine($"  Checkpoint saved: {path}");
    }

    private static string FormatMetrics(Dictionary<string, float> metrics)
    {
        if (metrics.Count == 0)
            return "no metrics yet";
        return string.Join(", ", metrics.Select(kv => $"{kv.Key}={kv.Value:F4}"));
    }
}
