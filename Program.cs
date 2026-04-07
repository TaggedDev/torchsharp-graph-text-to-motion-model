using Microsoft.Extensions.Configuration;

internal class Program
{
    private static void Main(string[] args)
    {
        string mode = args.Length > 0 ? args[0] : string.Empty;
        if (!ValidateMode(mode))
            return;

        var optionOverrides = ParseModeOptions(mode, args.Skip(1).ToArray());
        string configFileName = mode == "clip-embedding" ? "preprocessing.json" : $"{mode}.json";
        var config = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile($"Configs/{configFileName}", optional: false, reloadOnChange: true)
            .AddInMemoryCollection(optionOverrides)
            .Build();

        Console.WriteLine($"Running mode={mode}");

        switch (mode)
        {
            case "preprocessing": Preprocess.Run(config); break;
            case "clip-embedding": Preprocess.RunClipOnly(config); break;
            case "training": Train.Run(config); break;
            case "inference": Inference.Run(config); break;
            default: throw new ArgumentException("Mode is invalid but should have been caught before.");
        }
    }

    private static bool ValidateMode(string s)
    {
        if (string.IsNullOrEmpty(s))
            throw new ArgumentException("Mode is required");
        return s switch
        {
            "preprocessing" or "clip-embedding" or "training" or "inference" => true,
            _ => throw new ArgumentException(
                $"Mode is invalid: {s}. Valid values are 'preprocessing', 'training', 'inference'.")
        };
    }

    private static Dictionary<string, string?> ParseModeOptions(string mode, string[] args)
    {
        var overrides = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        if (args.Length == 0)
            return overrides;

        if (mode != "training")
            throw new ArgumentException($"Mode '{mode}' does not accept additional arguments.");

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--resume":
                    if (i + 1 >= args.Length)
                        throw new ArgumentException("--resume requires a checkpoint path.");
                    overrides["ResumeCheckpoint"] = args[++i];
                    break;
                default:
                    throw new ArgumentException($"Unknown training argument: {args[i]}");
            }
        }

        return overrides;
    }
}
