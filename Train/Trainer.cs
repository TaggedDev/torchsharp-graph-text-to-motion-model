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
        _checkpointDir = config["CheckpointDir"] ?? "checkpoints";
        var deviceName = config["Device"] ?? "cuda";

        int numTimesteps = int.Parse(config["NumTimesteps"] ?? "1000");
        int nodeHidden = int.Parse(config["NodeHidden"] ?? "64");
        int numGcnLayers = int.Parse(config["NumGcnLayers"] ?? "4");
        float lr = float.Parse(config["LearningRate"] ?? "0.0001", CultureInfo.InvariantCulture);

        _device = new Device(deviceName);

        _trainSet = new MotionDataset(Path.Combine(processedPath, "train"), processedPath);
        _valSet = new MotionDataset(Path.Combine(processedPath, "val"), processedPath);
        _testSet = new MotionDataset(Path.Combine(processedPath, "test"), processedPath);

        Console.WriteLine($"Dataset: train={_trainSet.Count}, val={_valSet.Count}, test={_testSet.Count}");

        _scheduler = new DdpmScheduler(numTimesteps);
        _scheduler.To(_device);

        _denoiser = new GraphDenoiser(numGcnLayers, nodeHidden);
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
        double epochLoss = 0;
        double totalBatchSeconds = 0;
        int step = 0;

        foreach (var indices in batches)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            using var scope = NewDisposeScope();
            using var batch = _trainSet.LoadBatch(indices, _device);

            int B = (int)batch.Motion.shape[0];
            var t = _scheduler.SampleTimesteps(B, _device);

            var noise = torch.randn_like(batch.Motion) * batch.Mask.unsqueeze(-1);
            var xt = _scheduler.QSample(batch.Motion, t, noise);
            var predicted = _denoiser.forward(xt, t, batch.Condition);
            var loss = _scheduler.Loss(predicted, noise, batch.Mask);

            _optimizer.zero_grad();
            loss.backward();
            nn.utils.clip_grad_norm_(_denoiser.parameters(), _gradClipNorm);
            _optimizer.step();

            float lossVal = loss.item<float>();
            epochLoss += lossVal;
            step++;

            sw.Stop();
            totalBatchSeconds += sw.Elapsed.TotalSeconds;

            if (step % _logEveryNSteps == 0)
            {
                double avgBatchSeconds = totalBatchSeconds / step;
                Console.WriteLine(
                    $"  Epoch {epoch} step {step}/{batches.Length} " +
                    $"loss={lossVal:F4} avg_batch_time={avgBatchSeconds:F4}s");
            }
        }

        return (float)(epochLoss / Math.Max(step, 1));
    }

    private (float avgLoss, Dictionary<string, float> metrics) ComputeMetrics(
        MotionDataset dataset)
    {
        _denoiser.eval();
        var batches = dataset.GetEpochBatches(_batchSize, shuffle: false);
        double totalLoss = 0;
        int count = 0;

        using (no_grad())
        {
            foreach (var indices in batches)
            {
                using var scope = NewDisposeScope();
                using var batch = dataset.LoadBatch(indices, _device);

                int B = (int)batch.Motion.shape[0];
                var t = _scheduler.SampleTimesteps(B, _device);

                var noise = randn_like(batch.Motion) * batch.Mask.unsqueeze(-1);
                var xt = _scheduler.QSample(batch.Motion, t, noise);
                var predicted = _denoiser.forward(xt, t, batch.Condition);
                var loss = _scheduler.Loss(predicted, noise, batch.Mask);

                totalLoss += loss.item<float>();
                count++;
            }
        }

        float avgLoss = (float)(totalLoss / Math.Max(count, 1));
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
        var path = Path.Combine(_checkpointDir, $"model_epoch{epoch}_{timestamp}.pt");
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
