using Microsoft.Extensions.Configuration;

internal class Program
{
    private static void Main(string[] args)
    {
        string mode = args.Length > 0 ? args[0] : string.Empty;
        if (!ValidateMode(mode))
            return;

        string configFileName = $"{mode}.json";
        var config = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile($"Configs/{configFileName}", optional: false, reloadOnChange: true)
            .Build();

        Console.WriteLine($"Running mode={mode}");

        switch (mode)
        {
            case "preprocessing": Preprocess.Run(config); break;
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
            "preprocessing" or "training" or "inference" => true,
            _ => throw new ArgumentException(
                $"Mode is invalid: {s}. Valid values are 'preprocessing', 'training', 'inference'.")
        };
    }
}