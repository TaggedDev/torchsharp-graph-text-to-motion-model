using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Extensions.Configuration;
using NumSharp;

public static class Preprocess
{
    private static readonly string[] Splits = ["train", "val", "test"];

    public static void Run(IConfigurationRoot config)
    {
        var datasetPath = config["DatasetPath"]
            ?? throw new InvalidOperationException("DatasetPath is not configured");
        var outputPath = config["OutputPath"]
            ?? throw new InvalidOperationException("OutputPath is not configured");

        Directory.CreateDirectory(outputPath);

        foreach (var split in Splits)
            ProcessSplit(datasetPath, outputPath, split);

        ConvertNormalizationStats(datasetPath, outputPath);

        Console.WriteLine("Preprocessing complete.");
    }

    private static void ProcessSplit(string datasetPath, string outputPath, string split)
    {
        var splitFile = Path.Combine(datasetPath, $"{split}.txt");
        if (!File.Exists(splitFile))
        {
            Console.WriteLine($"Skipping {split}: {splitFile} not found");
            return;
        }

        var ids = File.ReadAllLines(splitFile)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .ToArray();

        var splitDir = Path.Combine(outputPath, split);
        Directory.CreateDirectory(splitDir);

        Console.WriteLine($"Processing {split}: {ids.Length} samples");

        var processed = 0;
        var skipped = 0;

        foreach (var id in ids)
        {
            var npyPath = Path.Combine(datasetPath, "new_joint_vecs", $"{id}.npy");
            var txtPath = Path.Combine(datasetPath, "texts", $"{id}.txt");

            if (!File.Exists(npyPath))
            {
                skipped++;
                continue;
            }

            var motionData = np.Load<float[,]>(npyPath);
            var frameCount = motionData.GetLength(0);
            var featureDim = motionData.GetLength(1);

            var caption = ParseCaption(txtPath);

            var binPath = Path.Combine(splitDir, $"{id}.bin");
            WriteSampleBin(binPath, motionData, frameCount, featureDim, caption);

            processed++;
            if (processed % 1000 == 0)
                Console.WriteLine($"  {split}: {processed}/{ids.Length}");
        }

        Console.WriteLine($"  {split}: done — {processed} written, {skipped} skipped");
    }

    private static string ParseCaption(string txtPath)
    {
        if (!File.Exists(txtPath))
            return "";

        var firstLine = File.ReadLines(txtPath).FirstOrDefault() ?? "";
        var hashIdx = firstLine.IndexOf('#');
        return hashIdx >= 0 ? firstLine[..hashIdx].Trim() : firstLine.Trim();
    }

    private static void WriteSampleBin(
        string path, float[,] motionData, int frameCount, int featureDim, string caption)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 65536);
        using var bw = new BinaryWriter(fs);

        bw.Write(frameCount);
        bw.Write(featureDim);

        var flat = MemoryMarshal.CreateReadOnlySpan(
            ref motionData[0, 0], frameCount * featureDim);
        var bytes = MemoryMarshal.AsBytes(flat);
        bw.Write(bytes);

        var captionBytes = Encoding.UTF8.GetBytes(caption);
        bw.Write(captionBytes.Length);
        bw.Write(captionBytes);
    }

    private static void ConvertNormalizationStats(string datasetPath, string outputPath)
    {
        ConvertStat(Path.Combine(datasetPath, "Mean.npy"), Path.Combine(outputPath, "mean.bin"));
        ConvertStat(Path.Combine(datasetPath, "Std.npy"), Path.Combine(outputPath, "std.bin"));
        Console.WriteLine("Normalization stats converted.");
    }

    private static void ConvertStat(string npyPath, string binPath)
    {
        var data = np.Load<float[]>(npyPath);
        using var fs = new FileStream(binPath, FileMode.Create);
        using var bw = new BinaryWriter(fs);
        var bytes = MemoryMarshal.AsBytes(data.AsSpan());
        bw.Write(bytes);
    }
}
