using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Options;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Text2Motion.TorchTrainer;

public class ModelCheckpointService(IOptions<TrainingSettings> options)
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private static readonly Regex CheckpointRegex = new(@"^checkpoint-epoch-(\d{4})\.pt$", RegexOptions.Compiled);

    private readonly TrainingSettings _settings = options.Value;

    public void SaveEpochCheckpoint(string checkpointsPath, Module<Tensor, Tensor> model, int epoch)
    {
        var checkpoint = new CheckpointMetadata
        {
            Epoch = epoch,
            LearningRate = _settings.LearningRate,
            WeightDecay = _settings.WeightDecay
        };

        string modelPath = Path.Combine(checkpointsPath, $"checkpoint-epoch-{epoch:0000}.pt");
        string metadataPath = Path.Combine(checkpointsPath, $"checkpoint-epoch-{epoch:0000}.json");

        model.save(modelPath);
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(checkpoint, JsonOptions));
    }

    public void SaveFinalArtifacts(string runDirectoryPath, string testMetricsPath,
        Module<Tensor, Tensor> model)
    {
        var checkpoint = new CheckpointMetadata
        {
            Epoch = -1,
            LearningRate = _settings.LearningRate,
            WeightDecay = _settings.WeightDecay
        };

        string modelPath = Path.Combine(runDirectoryPath, "checkpoint-final.pt");
        string metadataPath = Path.Combine(runDirectoryPath, "checkpoint-final.json");

        model.save(modelPath);
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(checkpoint, JsonOptions));
    }

    public int RestoreCheckpoint(string runDirectoryPath, Module<Tensor, Tensor> model)
    {
        string checkpointsPath = Path.Combine(runDirectoryPath, "checkpoints");
        if (!Directory.Exists(checkpointsPath))
            throw new InvalidOperationException($"Checkpoint directory does not exist: {checkpointsPath}");

        int latestEpoch = Directory.GetFiles(checkpointsPath, "checkpoint-epoch-*.pt")
            .Select(Path.GetFileName)
            .Select(TryParseCheckpointEpoch)
            .Where(epoch => epoch.HasValue)
            .Select(epoch => epoch!.Value)
            .DefaultIfEmpty(0)
            .Max();

        if (latestEpoch == 0)
            throw new InvalidOperationException($"No model checkpoints found in {checkpointsPath}");

        string modelPath = Path.Combine(checkpointsPath, $"checkpoint-epoch-{latestEpoch:0000}.pt");
        string metadataPath = Path.Combine(checkpointsPath, $"checkpoint-epoch-{latestEpoch:0000}.json");

        if (!File.Exists(metadataPath))
            throw new InvalidOperationException($"Checkpoint metadata does not exist: {metadataPath}");

        model.load(modelPath);
        _ = JsonSerializer.Deserialize<CheckpointMetadata>(File.ReadAllText(metadataPath), JsonOptions)
            ?? throw new InvalidOperationException($"Checkpoint metadata is invalid: {metadataPath}");


        Console.WriteLine($"Resumed training from Run-{Path.GetFileName(runDirectoryPath).Split('-').Last()} epoch {latestEpoch:0000}.");
        return latestEpoch;
    }

    private static int? TryParseCheckpointEpoch(string? fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            return null;

        var match = CheckpointRegex.Match(fileName);
        return match.Success && int.TryParse(match.Groups[1].Value, out int epoch) ? epoch : null;
    }

    private sealed class CheckpointMetadata
    {
        public int Epoch { get; set; }
        public double LearningRate { get; set; }
        public double WeightDecay { get; set; }
    }
}
