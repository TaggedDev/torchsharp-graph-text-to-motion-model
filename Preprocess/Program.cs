using Microsoft.Extensions.Configuration;

public static class Preprocess
{
    public static void Run(IConfigurationRoot config)
    {
        Console.WriteLine("Running preprocessing...");
    }
}
