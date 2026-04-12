using System.Net;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Text2Motion.ClipModel;

public class ClipModelOnnxInference
{
    private const string ModelFileName = "clip-text-vit-32-float32-int32.onnx";
    private const string ModelUrl = "https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-text-vit-32-float32-int32.onnx";
    
    private readonly InferenceSession _session;
    private readonly ClipTokenizer _tokenizer;

    public ClipModelOnnxInference()
    {
        // Navigate from bin/Debug/net8.0 up to project root, then to Weights/Clip
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        string projectRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", ".."));
        string weightsDir = Path.Combine(projectRoot, "Weights", "Clip");
        Directory.CreateDirectory(weightsDir);

        string modelPath = Path.Combine(weightsDir, ModelFileName);

        if (!File.Exists(modelPath))
        {
            using var webClient = new WebClient();
            Console.WriteLine($"Not found \"{modelPath}\"...\n\t=> Downloading {modelPath}");
            webClient.DownloadFile(ModelUrl, modelPath);
        }

        _tokenizer = new ClipTokenizer();
        _session = new InferenceSession(modelPath);
    }

    public float[] GetTextEmbedding(string text)
    {
        if (_session == null)
            throw new InvalidOperationException("Model not initialized");

        // Tokenize text to token IDs
        int[] tokenIds = TokenizeText(text);

        // Create input tensor with shape [1, sequence_length]
        var inputTensor = new DenseTensor<int>(tokenIds, new[] { 1, tokenIds.Length });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        // Run inference
        using var results = _session.Run(inputs);

        // Extract embeddings from output
        var output = results.FirstOrDefault(r => r.Name == "output");
        if (output == null)
            throw new InvalidOperationException("Model output not found");

        // Convert to float array
        var embeddings = (output.Value as IEnumerable<float>)?.ToArray()
                         ?? throw new InvalidOperationException("Failed to extract embeddings");

        return embeddings;
    }

    private int[] TokenizeText(string text) 
        => _tokenizer.Tokenize(text);
}