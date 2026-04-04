using Microsoft.Extensions.Configuration;
using NumSharp;

public static class Preprocess
{
    public static void Run(IConfigurationRoot config)
    {
        var datasetPath = config["DatasetPath"]
            ?? throw new InvalidOperationException("DatasetPath is not configured");

        var firstFile = Directory.EnumerateFiles(Path.Combine(datasetPath, "new_joint_vecs"), "*.npy")
            .Order()
            .First();

        var data = np.Load<float[,]>(firstFile);
        Console.WriteLine($"Loaded {firstFile}: shape=({data.GetLength(0)}, {data.GetLength(1)})");
        Console.WriteLine($"First frame (first 5 values): [{string.Join(", ", Enumerable.Range(0, 5).Select(i => data[0, i]))}]");
    }
}
