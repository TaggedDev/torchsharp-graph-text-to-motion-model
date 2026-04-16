using Microsoft.Extensions.Options;
using ShellProgressBar;
using Text2Motion.ClipModel;

namespace Text2Motion.DataPreprocessing;

public class DataPreprocessor(ClipModelOnnxInference clipModel, IOptions<PreprocessingConfig> options)
{
    private readonly PreprocessingConfig _config = options.Value;

    public async Task RunAsync(CancellationToken token)
    {
        if (await CheckPreprocessingsExistAsync(token))
            return;

        var files = Directory.GetFiles(_config.AnnotationPath, "*.txt");
        int total = files.Length;

        var options = new ProgressBarOptions
        {
            ForegroundColor = ConsoleColor.Cyan,
            BackgroundColor = ConsoleColor.DarkGray,
            ProgressCharacter = '─',
            DisplayTimeInRealTime = false,
        };

        using var progressBar = new ProgressBar(total, "Preprocessing embeddings", options);

        for (int i = 0; i < total; i++)
        {
            token.ThrowIfCancellationRequested();
            string file = files[i];
            string name = Path.GetFileNameWithoutExtension(file);

            progressBar.Tick($"[{i + 1}/{total} ({(i + 1) * 100 / total}%)] {name}");

            string text = await GetFirstAnnotationAsync(file, token);
            float[] embedding = clipModel.GetTextEmbedding(text);
            await SaveEmbeddingFileAsync(embedding, name, token);
        }
    }

    private Task<bool> CheckPreprocessingsExistAsync(CancellationToken token)
    {
        bool exists = Directory.Exists(_config.EmbeddingsPath)
            && Directory.GetFiles(_config.EmbeddingsPath, "*.bin").Length > 0;
        return Task.FromResult(exists);
    }

    private async Task SaveEmbeddingFileAsync(float[] embedding, string sourceFilePath, CancellationToken token)
    {
        Directory.CreateDirectory(_config.EmbeddingsPath);

        string name = Path.GetFileNameWithoutExtension(sourceFilePath);
        string outputPath = Path.Combine(_config.EmbeddingsPath, $"{name}.bin");

        byte[] bytes = new byte[embedding.Length * sizeof(float)];
        Buffer.BlockCopy(embedding, 0, bytes, 0, bytes.Length);
        await File.WriteAllBytesAsync(outputPath, bytes, token);
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
}