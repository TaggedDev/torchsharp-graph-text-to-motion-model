using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Options;
using Text2Motion.ClipModel;

namespace Text2Motion.DataPreprocessing;

public class DataPreprocessor(ClipModelOnnxInference clipModel, IOptions<PreprocessingConfig> options)
{
    private readonly ClipModelOnnxInference _clipModel = clipModel;
    private readonly PreprocessingConfig _config = options.Value;

    public async Task RunAsync(CancellationToken token)
    {
        if (await CheckPreprocessingsExistAsync(token))
            return;

        string[] annotations = Directory.GetFiles(_config.AnnotationPath, "*.txt");
        foreach (var annotationFile in annotations)
        {
            string text = await GetFirstAnnotationAsync(annotationFile, token);
            float[] embedding = _clipModel.GetTextEmbedding(text);
            await SaveEmbeddingFileAsync(embedding, text, token, annotationFile);
        }
    }

    private async Task<bool> CheckPreprocessingsExistAsync(CancellationToken token)
    {
        if (!Directory.Exists(_config.EmbeddingsPath))
            return false;

        var files = await Task.Run(() => Directory.GetFiles(_config.EmbeddingsPath, "*.txt"), token);
        return files.Length > 0;
    }

    private async Task SaveEmbeddingFileAsync(float[] embedding, string text, CancellationToken token, string fileName)
    {
        Directory.CreateDirectory(_config.EmbeddingsPath);

        string outputPath = Path.Combine(_config.EmbeddingsPath, $"{fileName}.txt");

        var embeddingData = new { embedding, text };
        string json = JsonSerializer.Serialize(embeddingData);
        await File.WriteAllTextAsync(outputPath, json, token);
    }

    private async Task<string> GetFirstAnnotationAsync(string fileName, CancellationToken token)
    {
        string content = await File.ReadAllTextAsync(fileName, token);
        string[] lines = content.Split(["\r\n", "\r", "\n"], StringSplitOptions.None);
        if (lines.Length == 0)
            return string.Empty;

        string firstLine = lines[0];
        int hashIndex = firstLine.IndexOf('#');
        return hashIndex >= 0 ? firstLine[..hashIndex].Trim() : firstLine.Trim();
    }

    private string ComputeHash(string text)
    {
        using var sha256 = SHA256.Create();
        byte[] hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(text));
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }
}