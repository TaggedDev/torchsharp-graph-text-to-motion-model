using Tokenizers.DotNet;

namespace Text2Motion.ClipModel;

public class ClipTokenizer
{
    private const string TokenizerJSONFilename = "tokenizer.json";
    private const string TokenizerFileFolder = "Weights/Clip";
    private const int MaxLength = 77; // CLIP's default context length

    private readonly Tokenizer _tokenizer;

    public ClipTokenizer()
    {
        string tokenizerPath = Path.Combine(TokenizerFileFolder, TokenizerJSONFilename);
        _tokenizer = new Tokenizer(tokenizerPath);
    }

    public int[] Tokenize(string text)
    {
        var tokens = _tokenizer.Encode(text).ToArray();
        var paddedTokens = new int[MaxLength];
        // Truncate if too long, else pad with zeros if too short (0 is the padding token)
        Array.Copy(tokens, paddedTokens, tokens.Length >= MaxLength ? MaxLength : tokens.Length);
        return paddedTokens;
    }
}