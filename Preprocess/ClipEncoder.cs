using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public sealed class ClipEncoder : IDisposable
{
    private readonly InferenceSession _session;
    private const int TokenLength = 77;
    private const int EmbeddingDim = 512;

    public ClipEncoder(string onnxPath, bool useGpu = true)
    {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        if (useGpu)
        {
            try
            {
                RegisterNativeLibraryPath();
                options.AppendExecutionProvider_CUDA(0);
                Console.WriteLine("CLIP encoder: using CUDA GPU");
            }
            catch
            {
                Console.WriteLine("CLIP encoder: CUDA unavailable, falling back to CPU");
            }
        }

        _session = new InferenceSession(onnxPath, options);
    }

    private static void RegisterNativeLibraryPath()
    {
        // The NuGet package places native DLLs (cuDNN, CUDA provider) under
        // runtimes/win-x64/native/. The CUDA provider loads cuDNN via LoadLibrary
        // which doesn't know about this subfolder. We prepend it to PATH so the
        // system loader can find cudnn_graph64_9.dll and other dependencies.
        var baseDir = AppContext.BaseDirectory;
        var nativePath = Path.Combine(baseDir, "runtimes", "win-x64", "native");

        if (Directory.Exists(nativePath))
        {
            var currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            if (!currentPath.Contains(nativePath, StringComparison.OrdinalIgnoreCase))
            {
                Environment.SetEnvironmentVariable("PATH", nativePath + ";" + currentPath);
            }
        }
    }

    public float[] Encode(long[] tokenIds)
    {
        if (tokenIds.Length != TokenLength)
            throw new ArgumentException($"Expected {TokenLength} tokens, got {tokenIds.Length}");

        var input = new DenseTensor<long>(tokenIds, [1, TokenLength]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", input)
        };

        using var results = _session.Run(inputs);
        var output = results[0].AsTensor<float>();
        var embedding = new float[EmbeddingDim];
        for (var i = 0; i < EmbeddingDim; i++)
            embedding[i] = output[0, i];

        return embedding;
    }

    public void Dispose() => _session.Dispose();
}
