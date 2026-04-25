using Microsoft.Extensions.Options;
using NumSharp;
using ShellProgressBar;
using TorchSharp;
using Text2Motion.TorchTrainer;
using static TorchSharp.torch;

namespace Text2Motion.Dataset;

public class HumanML3DDataset
{
    private readonly DatasetSettings _settings;
    private float[] _mean = Array.Empty<float>();
    private float[] _std = Array.Empty<float>();
    private List<MotionSample> _trainSamples = new();
    private List<MotionSample> _valSamples = new();
    private List<MotionSample> _testSamples = new();

    public IReadOnlyList<MotionSample> Train => _trainSamples.AsReadOnly();
    public IReadOnlyList<MotionSample> Val => _valSamples.AsReadOnly();
    public IReadOnlyList<MotionSample> Test => _testSamples.AsReadOnly();

    public HumanML3DDataset(IOptions<DatasetSettings> options)
    {
        _settings = options.Value;
    }

    public async Task LoadAsync()
    {
        LoadNormalizationStats();

        if (!string.IsNullOrWhiteSpace(_settings.TrainSplitPath))
        {
            var ids = await ReadSplitFile(_settings.TrainSplitPath);
            _trainSamples = await LoadSamplesAsync(ids, "train");
            Console.WriteLine($"Loaded {_trainSamples.Count} training samples.");
        }

        if (!string.IsNullOrWhiteSpace(_settings.ValidationSplitPath))
        {
            var ids = await ReadSplitFile(_settings.ValidationSplitPath);
            _valSamples = await LoadSamplesAsync(ids, "val");
            Console.WriteLine($"Loaded {_valSamples.Count} validation samples.");
        }

        if (!string.IsNullOrWhiteSpace(_settings.TestSplitPath))
        {
            var ids = await ReadSplitFile(_settings.TestSplitPath);
            _testSamples = await LoadSamplesAsync(ids, "test");
            Console.WriteLine($"Loaded {_testSamples.Count} test samples.");
        }
    }

    private void LoadNormalizationStats()
    {
        if (string.IsNullOrWhiteSpace(_settings.NormalizationMeanPath) ||
            string.IsNullOrWhiteSpace(_settings.NormalizationStdPath))
        {
            throw new InvalidOperationException("Normalization paths not configured.");
        }

        try
        {
            _mean = LoadNpyAsFloat32(_settings.NormalizationMeanPath);
            _std = LoadNpyAsFloat32(_settings.NormalizationStdPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to load normalization stats from {_settings.NormalizationMeanPath} and {_settings.NormalizationStdPath}", ex);
        }
    }

    private async Task<List<string>> ReadSplitFile(string splitPath)
    {
        var lines = await File.ReadAllLinesAsync(splitPath);
        return lines
            .Select(line => line.Trim())
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .ToList();
    }

    private async Task<List<MotionSample>> LoadSamplesAsync(List<string> ids, string label)
    {
        var samples = new List<MotionSample>();
        int skipped = 0;

        var barOptions = new ProgressBarOptions
        {
            ForegroundColor = ConsoleColor.Cyan,
            BackgroundColor = ConsoleColor.DarkGray,
            ProgressCharacter = '─',
            DisplayTimeInRealTime = false,
        };
        using var bar = new ProgressBar(ids.Count, $"Loading {label} samples", barOptions);

        foreach (var id in ids)
        {
            try
            {
                var motionPath = Path.Combine(_settings.JointsPath, $"{id}.npy");
                var embeddingPath = Path.Combine(_settings.EmbeddingsPath, $"{id}.bin");

                if (!File.Exists(motionPath) || !File.Exists(embeddingPath))
                {
                    skipped++;
                    bar.Tick(id);
                    continue;
                }

                var textEmbedding = LoadTextEmbedding(embeddingPath);
                var motionFrames = LoadAndNormalizeMotion(motionPath);

                if (textEmbedding.Length > 0 && motionFrames.Length > 0)
                {
                    samples.Add(new MotionSample(textEmbedding, motionFrames));
                }
                bar.Tick(id);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to load sample {id}: {ex.Message}");
                skipped++;
                bar.Tick(id);
            }
        }

        if (skipped > 0)
        {
            Console.WriteLine($"  Skipped {skipped} samples due to missing files or errors.");
        }

        return samples;
    }

    private float[] LoadTextEmbedding(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    private float[] LoadAndNormalizeMotion(string path)
    {
        try
        {
            var flatArray = LoadNpyAsFloat32(path);

            if (flatArray.Length != _settings.FixedFrames * _settings.FeatureDim)
            {
                flatArray = NormalizeAndPadOrCrop(flatArray);
            }
            else
            {
                flatArray = Normalize(flatArray);
            }

            return flatArray;
        }
        catch
        {
            return Array.Empty<float>();
        }
    }

    private static float[] LoadNpyAsFloat32(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // Read magic number: b'\x93NUMPY'
        var magic = reader.ReadBytes(6);
        if (magic[0] != 0x93 || magic[1] != (byte)'N' || magic[2] != (byte)'U' ||
            magic[3] != (byte)'M' || magic[4] != (byte)'P' || magic[5] != (byte)'Y')
        {
            throw new InvalidOperationException("Not a valid .npy file");
        }

        // Version (1 byte major, 1 byte minor)
        byte major = reader.ReadByte();
        byte minor = reader.ReadByte();

        // Header length
        ushort headerLen = reader.ReadUInt16();

        // Read header
        var headerBytes = reader.ReadBytes(headerLen);
        var headerStr = System.Text.Encoding.UTF8.GetString(headerBytes);

        // Parse dtype and shape from header (simplified; assumes float32)
        var floatData = new List<float>();
        while (stream.Position < stream.Length)
        {
            try
            {
                floatData.Add(reader.ReadSingle());
            }
            catch
            {
                break;
            }
        }

        return floatData.ToArray();
    }

    private float[] NormalizeAndPadOrCrop(float[] data)
    {
        int featureDim = _settings.FeatureDim;
        int fixedFrames = _settings.FixedFrames;
        int targetSize = fixedFrames * featureDim;

        if (data.Length < targetSize)
        {
            var padded = new float[targetSize];
            Array.Copy(data, padded, data.Length);
            return padded;
        }
        else if (data.Length > targetSize)
        {
            var cropped = new float[targetSize];
            Array.Copy(data, cropped, targetSize);
            return cropped;
        }

        return data;
    }

    private float[] Normalize(float[] data)
    {
        if (_mean.Length != data.Length || _std.Length != data.Length)
            return data;

        var normalized = new float[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            normalized[i] = _std[i] > 1e-8f ? (data[i] - _mean[i]) / _std[i] : (data[i] - _mean[i]);
        }
        return normalized;
    }

    public (Tensor textEmbeddings, Tensor motionFrames) GetBatch(
        IReadOnlyList<MotionSample> samples,
        IEnumerable<int> indices,
        Device device)
    {
        var indexList = indices.ToList();
        int batchSize = indexList.Count;
        var textEmbList = new List<float[]>();
        var motionFramesList = new List<float[]>();

        foreach (var idx in indexList)
        {
            textEmbList.Add(samples[idx].TextEmbedding);
            motionFramesList.Add(samples[idx].MotionFrames);
        }

        var textEmbTensor = CreateTensor(textEmbList, device);
        var motionTensor = CreateTensor(motionFramesList, device);

        return (textEmbTensor, motionTensor);
    }

    private static Tensor CreateTensor(List<float[]> data, Device device)
    {
        int batchSize = data.Count;
        int featureSize = data[0].Length;

        var flatArray = new float[batchSize * featureSize];
        for (int i = 0; i < batchSize; i++)
        {
            Array.Copy(data[i], 0, flatArray, i * featureSize, featureSize);
        }

        var tensor = from_array(flatArray);
        tensor = tensor.reshape(batchSize, featureSize);
        return tensor.to(device);
    }
}
