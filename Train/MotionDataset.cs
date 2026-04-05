using static TorchSharp.torch;

namespace Training;

public record MotionBatch(
    Tensor Motion,     // [B, Tmax, 263] normalized, zero-padded
    Tensor Mask,       // [B, Tmax]      1.0 = real frame, 0.0 = padding
    Tensor Condition   // [B, 512]       CLIP embedding (one per sample)
) : IDisposable
{
    public void Dispose()
    {
        Motion.Dispose();
        Mask.Dispose();
        Condition.Dispose();
    }
}

public sealed class MotionDataset
{
    private readonly List<(float[] motion, int frames, float[][] clips)> _samples = [];
    private readonly Random _rng = new(42);

    public int Count => _samples.Count;

    public MotionDataset(string splitDir, string processedRoot)
    {
        var mean = ReadRawFloats(Path.Combine(processedRoot, "mean.bin"), 263);
        var std = ReadRawFloats(Path.Combine(processedRoot, "std.bin"), 263);

        var binFiles = Directory.GetFiles(splitDir, "*.bin")
            .Where(f => !f.EndsWith(".clip.bin", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f)
            .ToArray();

        foreach (var binPath in binFiles)
        {
            var clipPath = Path.ChangeExtension(binPath, ".clip.bin");
            if (!File.Exists(clipPath))
                continue;

            var (data, frames, featDim) = ReadMotion(binPath);
            var clips = ReadClip(clipPath);

            // Normalize in-place during load
            for (int t = 0; t < frames; t++)
            {
                int off = t * featDim;
                for (int f = 0; f < featDim; f++)
                {
                    float s = std[f];
                    data[off + f] = (data[off + f] - mean[f]) / (s < 1e-5f ? 1.0f : s);
                }
            }

            _samples.Add((data, frames, clips));
        }

        Console.WriteLine($"  Loaded {_samples.Count} samples from {splitDir}");
    }

    public int[][] GetEpochBatches(int batchSize, bool shuffle)
    {
        var indices = Enumerable.Range(0, _samples.Count).ToArray();
        if (shuffle)
        {
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = _rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        var batches = new List<int[]>();
        for (int i = 0; i < indices.Length; i += batchSize)
        {
            int len = Math.Min(batchSize, indices.Length - i);
            var batch = new int[len];
            Array.Copy(indices, i, batch, 0, len);
            batches.Add(batch);
        }
        return batches.ToArray();
    }

    public MotionBatch LoadBatch(int[] indices, Device device)
    {
        int B = indices.Length;
        int featDim = Data.Skeleton.FeatureDim;

        int tMax = 0;
        for (int i = 0; i < B; i++)
        {
            int frames = _samples[indices[i]].frames;
            if (frames > tMax) tMax = frames;
        }

        var motionBuf = new float[B * tMax * featDim];
        var maskBuf = new float[B * tMax];
        var condBuf = new float[B * 512];

        for (int i = 0; i < B; i++)
        {
            var (data, frames, clips) = _samples[indices[i]];

            // Copy pre-normalized motion data
            int srcLen = frames * featDim;
            int dstOff = i * tMax * featDim;
            Array.Copy(data, 0, motionBuf, dstOff, srcLen);

            // Fill mask
            int maskOff = i * tMax;
            for (int t = 0; t < frames; t++)
                maskBuf[maskOff + t] = 1.0f;

            // Pick a random CLIP embedding
            var clip = clips[_rng.Next(clips.Length)];
            Array.Copy(clip, 0, condBuf, i * 512, 512);
        }

        var motionT = tensor(motionBuf, [B, tMax, featDim], dtype: float32).to(device);
        var maskT = tensor(maskBuf, [B, tMax], dtype: float32).to(device);
        var condT = tensor(condBuf, [B, 512], dtype: float32).to(device);

        return new MotionBatch(motionT, maskT, condT);
    }

    private static (float[] data, int frames, int featDim) ReadMotion(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
        using var br = new BinaryReader(fs);

        int frames = br.ReadInt32();
        int featDim = br.ReadInt32();
        int count = frames * featDim;
        var bytes = br.ReadBytes(count * sizeof(float));
        var data = new float[count];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return (data, frames, featDim);
    }

    private static float[][] ReadClip(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
        using var br = new BinaryReader(fs);

        int numCaptions = br.ReadInt32();
        var result = new float[numCaptions][];

        for (int i = 0; i < numCaptions; i++)
        {
            int dim = br.ReadInt32();
            var bytes = br.ReadBytes(dim * sizeof(float));
            var emb = new float[dim];
            Buffer.BlockCopy(bytes, 0, emb, 0, bytes.Length);
            result[i] = emb;
        }
        return result;
    }

    private static float[] ReadRawFloats(string path, int expectedCount)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[expectedCount];
        Buffer.BlockCopy(bytes, 0, floats, 0, expectedCount * sizeof(float));
        return floats;
    }
}
