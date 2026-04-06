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

    public MotionBatch LoadBatch(int[] indices, Device device, float condDropoutProb = 0f, int maxSequenceLength = 0)
    {
        int B = indices.Length;
        int featDim = Data.Skeleton.FeatureDim;

        // Per-sample effective frame count after optional random crop to
        // maxSequenceLength. Crop offsets are picked once here so we can
        // both size the batch tensor and copy the right window.
        var effFrames = new int[B];
        var cropOffsets = new int[B];
        int tMax = 0;
        for (int i = 0; i < B; i++)
        {
            int frames = _samples[indices[i]].frames;
            int eff = frames;
            int offset = 0;
            if (maxSequenceLength > 0 && frames > maxSequenceLength)
            {
                eff = maxSequenceLength;
                offset = _rng.Next(frames - maxSequenceLength + 1);
            }
            effFrames[i] = eff;
            cropOffsets[i] = offset;
            if (eff > tMax) tMax = eff;
        }

        var motionBuf = new float[B * tMax * featDim];
        var maskBuf = new float[B * tMax];
        var condBuf = new float[B * 512];

        for (int i = 0; i < B; i++)
        {
            var (data, _, clips) = _samples[indices[i]];
            int eff = effFrames[i];
            int srcFrameOff = cropOffsets[i];

            // Copy pre-normalized motion data (cropped window)
            int srcLen = eff * featDim;
            int srcByteOff = srcFrameOff * featDim;
            int dstOff = i * tMax * featDim;
            Array.Copy(data, srcByteOff, motionBuf, dstOff, srcLen);

            // Fill mask
            int maskOff = i * tMax;
            for (int t = 0; t < eff; t++)
                maskBuf[maskOff + t] = 1.0f;

            // Classifier-free guidance: with probability condDropoutProb drop
            // the CLIP embedding (leave zeros) so the model learns both
            // conditional and unconditional noise prediction.
            if (condDropoutProb > 0f && _rng.NextDouble() < condDropoutProb)
            {
                // condBuf slice is already zero-initialized — leave as-is.
            }
            else
            {
                var clip = clips[_rng.Next(clips.Length)];
                Array.Copy(clip, 0, condBuf, i * 512, 512);
            }
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
