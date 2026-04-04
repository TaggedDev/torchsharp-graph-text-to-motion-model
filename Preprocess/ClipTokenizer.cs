using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

public sealed partial class ClipTokenizer
{
    private const int ContextLength = 77;
    private const int SotToken = 49406; // <|startoftext|>
    private const int EotToken = 49407; // <|endoftext|>

    private readonly Dictionary<string, int> _encoder;
    private readonly Dictionary<(string, string), int> _bpeRanks;
    private readonly Dictionary<int, char> _byteToUnicode;

    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", RegexOptions.IgnoreCase)]
    private static partial Regex TokenPattern();

    public ClipTokenizer(string vocabGzPath)
    {
        _byteToUnicode = BuildByteToUnicode();

        var lines = ReadGzipLines(vocabGzPath);

        // Lines 1..48894 are BPE merge pairs (line 0 is a header)
        var merges = lines.Skip(1).Take(48894).ToList();
        _bpeRanks = new Dictionary<(string, string), int>(merges.Count);
        for (var i = 0; i < merges.Count; i++)
        {
            var parts = merges[i].Split(' ');
            _bpeRanks[(parts[0], parts[1])] = i;
        }

        // Build encoder vocabulary
        var vocab = new List<string>();

        // 256 byte-level tokens
        foreach (var c in _byteToUnicode.OrderBy(kv => kv.Key).Select(kv => kv.Value))
            vocab.Add(c.ToString());

        // 256 byte-level tokens with </w> suffix
        foreach (var c in _byteToUnicode.OrderBy(kv => kv.Key).Select(kv => kv.Value))
            vocab.Add(c + "</w>");

        // Merge tokens
        foreach (var merge in merges)
        {
            var parts = merge.Split(' ');
            vocab.Add(parts[0] + parts[1]);
        }

        // Special tokens
        vocab.Add("<|startoftext|>");
        vocab.Add("<|endoftext|>");

        _encoder = new Dictionary<string, int>(vocab.Count);
        for (var i = 0; i < vocab.Count; i++)
            _encoder[vocab[i]] = i;
    }

    public long[] Encode(string text)
    {
        text = text.ToLowerInvariant().Trim();

        var tokens = new List<int> { SotToken };

        foreach (Match match in TokenPattern().Matches(text))
        {
            var word = match.Value;
            var encoded = EncodeWord(word);
            tokens.AddRange(encoded);
        }

        tokens.Add(EotToken);

        var result = new long[ContextLength];
        var len = Math.Min(tokens.Count, ContextLength);
        for (var i = 0; i < len; i++)
            result[i] = tokens[i];

        return result;
    }

    private List<int> EncodeWord(string word)
    {
        // Convert word to byte-level unicode representation
        var bytes = Encoding.UTF8.GetBytes(word);
        var sb = new StringBuilder(bytes.Length + 4);
        foreach (var b in bytes)
            sb.Append(_byteToUnicode[b]);

        var wordStr = sb.ToString();
        if (wordStr.Length == 0)
            return [];

        // Split into individual characters, append </w> to last
        var bpeTokens = new List<string>(wordStr.Length);
        for (var i = 0; i < wordStr.Length - 1; i++)
            bpeTokens.Add(wordStr[i].ToString());
        bpeTokens.Add(wordStr[^1] + "</w>");

        bpeTokens = ApplyBpe(bpeTokens);

        var ids = new List<int>(bpeTokens.Count);
        foreach (var token in bpeTokens)
        {
            if (_encoder.TryGetValue(token, out var id))
                ids.Add(id);
        }

        return ids;
    }

    private List<string> ApplyBpe(List<string> tokens)
    {
        if (tokens.Count <= 1)
            return tokens;

        while (true)
        {
            var bestRank = int.MaxValue;
            var bestIdx = -1;

            for (var i = 0; i < tokens.Count - 1; i++)
            {
                if (_bpeRanks.TryGetValue((tokens[i], tokens[i + 1]), out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0)
                break;

            var merged = tokens[bestIdx] + tokens[bestIdx + 1];
            var newTokens = new List<string>(tokens.Count - 1);
            for (var i = 0; i < tokens.Count; i++)
            {
                if (i == bestIdx)
                {
                    newTokens.Add(merged);
                    i++; // skip next
                }
                else
                {
                    newTokens.Add(tokens[i]);
                }
            }

            tokens = newTokens;

            if (tokens.Count <= 1)
                break;
        }

        return tokens;
    }

    /// <summary>
    /// Replicates CLIP's bytes_to_unicode(): maps byte values 0-255 to unicode characters.
    /// Printable bytes map to themselves, non-printable bytes map to chars starting at U+0100.
    /// </summary>
    private static Dictionary<int, char> BuildByteToUnicode()
    {
        var bs = new List<int>();
        var cs = new List<int>();

        // Printable ranges: '!' to '~', '¡' to '¬', '®' to 'ÿ'
        for (var i = '!'; i <= '~'; i++) { bs.Add(i); cs.Add(i); }
        for (var i = 0xA1; i <= 0xAC; i++) { bs.Add(i); cs.Add(i); }
        for (var i = 0xAE; i <= 0xFF; i++) { bs.Add(i); cs.Add(i); }

        // Non-printable bytes get mapped to chars >= 256
        var n = 0;
        for (var b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var result = new Dictionary<int, char>(256);
        for (var i = 0; i < 256; i++)
            result[bs[i]] = (char)cs[i];

        return result;
    }

    private static List<string> ReadGzipLines(string path)
    {
        using var fs = File.OpenRead(path);
        using var gz = new GZipStream(fs, CompressionMode.Decompress);
        using var reader = new StreamReader(gz, Encoding.UTF8);

        var lines = new List<string>();
        while (reader.ReadLine() is { } line)
            lines.Add(line);
        return lines;
    }
}
