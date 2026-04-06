using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Extensions.Configuration;
using TorchSharp;
using Training;
using static TorchSharp.torch;

/// <summary>
/// End-to-end text→motion inference: loads a GraphDenoiser checkpoint, a CLIP
/// text encoder, and normalization statistics, then generates motion .bin
/// files from a text prompt via reverse DDPM sampling.
/// Constructor is expensive (model + CLIP load). Cache and reuse per checkpoint.
/// </summary>
public sealed class InferenceEngine : IDisposable
{
    private const int FeatureDim = 263;
    private const int ClipEmbeddingDim = 512;
    private const int MaxFilenamePromptLen = 20;

    private readonly GraphDenoiser _denoiser;
    private readonly DdpmScheduler _scheduler;
    private readonly ClipTokenizer _tokenizer;
    private readonly ClipEncoder _clipEncoder;
    private readonly Device _device;
    private readonly float[] _mean;
    private readonly float[] _std;
    private readonly string _checkpointName;

    public string CheckpointName => _checkpointName;

    public InferenceEngine(
        string checkpointPath,
        string trainingConfigPath,
        string preprocessingConfigPath,
        string meanBinPath,
        string stdBinPath)
    {
        var trainingCfg = new ConfigurationBuilder()
            .AddJsonFile(trainingConfigPath, optional: false)
            .Build();

        int numTimesteps = int.Parse(trainingCfg["NumTimesteps"] ?? "1000");
        int nodeHidden = int.Parse(trainingCfg["NodeHidden"] ?? "64");
        int numGcnLayers = int.Parse(trainingCfg["NumGcnLayers"] ?? "4");
        var requestedDevice = trainingCfg["Device"] ?? "cuda";

        _device = (requestedDevice == "cuda" && cuda.is_available())
            ? CUDA
            : CPU;
        Console.WriteLine($"InferenceEngine: device={_device.type}");

        _denoiser = new GraphDenoiser(numGcnLayers, nodeHidden);
        _denoiser.load(checkpointPath);
        _denoiser.to(_device);
        _denoiser.MoveGcnBuffers(_device);
        _denoiser.eval();

        _scheduler = new DdpmScheduler(numTimesteps);
        _scheduler.To(_device);

        var preprocessingCfg = new ConfigurationBuilder()
            .AddJsonFile(preprocessingConfigPath, optional: false)
            .Build();

        var clipModelPath = preprocessingCfg["ClipModelPath"]
            ?? throw new InvalidOperationException("ClipModelPath missing from preprocessing config");
        var clipVocabPath = preprocessingCfg["ClipVocabPath"]
            ?? throw new InvalidOperationException("ClipVocabPath missing from preprocessing config");

        _tokenizer = new ClipTokenizer(clipVocabPath);
        _clipEncoder = new ClipEncoder(clipModelPath, useGpu: true);

        _mean = ReadRawFloats(meanBinPath, FeatureDim);
        _std = ReadRawFloats(stdBinPath, FeatureDim);

        _checkpointName = Path.GetFileNameWithoutExtension(checkpointPath);
    }

    /// <summary>
    /// Generate a motion .bin file for the given prompt. Returns the written file path.
    /// </summary>
    public string Generate(
        string prompt,
        int frames,
        string outputRoot,
        long? seed = null)
    {
        if (string.IsNullOrWhiteSpace(prompt))
            throw new ArgumentException("Prompt must be non-empty", nameof(prompt));

        frames = Math.Clamp(frames, 1, 300);

        // 1. Text → CLIP embedding [1, 512]
        var tokens = _tokenizer.Encode(prompt);
        var condVec = _clipEncoder.Encode(tokens);

        using var scope = NewDisposeScope();
        using var cond = tensor(condVec, new long[] { 1, ClipEmbeddingDim }, dtype: float32).to(_device);

        // 2. Reverse-sample in normalized feature space
        var generated = _scheduler.Sample(_denoiser, cond, frames, FeatureDim, _device, seed);

        // 3. Move to CPU and denormalize
        using var cpu = generated.to(CPU).contiguous();
        var data = cpu.data<float>().ToArray();

        for (int t = 0; t < frames; t++)
        {
            int off = t * FeatureDim;
            for (int f = 0; f < FeatureDim; f++)
            {
                float s = _std[f] < 1e-5f ? 1f : _std[f];
                data[off + f] = data[off + f] * s + _mean[f];
            }
        }

        if (!TryFindFirstNonFinite(data, out var badIndex, out var badValue))
        {
            int badFrame = badIndex / FeatureDim;
            int badFeature = badIndex % FeatureDim;
            throw new InvalidOperationException(
                $"Generated motion contains a non-finite value ({badValue}) at frame {badFrame}, feature {badFeature}. The checkpoint output is numerically unstable.");
        }

        // 4. Build output path: generations/{model}/{prompt_20}_{HH-mm-ss}.bin
        var sanitized = SanitizeForFilename(prompt, MaxFilenamePromptLen);
        var time = DateTime.Now.ToString("HH-mm-ss", CultureInfo.InvariantCulture);
        var dir = Path.Combine(outputRoot, _checkpointName);
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, $"{sanitized}_{time}.bin");

        // 5. Write .bin in the exact training/preprocess format
        WriteBin(path, frames, FeatureDim, data, prompt);

        return path;
    }

    private static void WriteBin(string path, int frames, int featureDim, float[] data, string caption)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 65536);
        using var bw = new BinaryWriter(fs);

        bw.Write(frames);
        bw.Write(featureDim);

        var bytes = MemoryMarshal.AsBytes(data.AsSpan());
        bw.Write(bytes);

        var captionBytes = Encoding.UTF8.GetBytes(caption);
        bw.Write(captionBytes.Length);
        bw.Write(captionBytes);
    }

    private static float[] ReadRawFloats(string path, int expectedCount)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[expectedCount];
        Buffer.BlockCopy(bytes, 0, floats, 0, expectedCount * sizeof(float));
        return floats;
    }

    private static bool TryFindFirstNonFinite(float[] data, out int index, out float value)
    {
        for (int i = 0; i < data.Length; i++)
        {
            var current = data[i];
            if (float.IsNaN(current) || float.IsInfinity(current))
            {
                index = i;
                value = current;
                return false;
            }
        }

        index = -1;
        value = 0f;
        return true;
    }

    private static string SanitizeForFilename(string s, int maxLen)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder();
        foreach (var c in s.Trim())
        {
            if (Array.IndexOf(invalid, c) >= 0 || char.IsControl(c))
                sb.Append('_');
            else if (char.IsWhiteSpace(c))
                sb.Append('_');
            else
                sb.Append(c);
        }

        var trimmed = sb.ToString().Trim('_');
        if (trimmed.Length == 0)
            trimmed = "prompt";
        if (trimmed.Length > maxLen)
            trimmed = trimmed[..maxLen];
        return trimmed;
    }

    public void Dispose()
    {
        _clipEncoder.Dispose();
        _denoiser.Dispose();
    }
}
