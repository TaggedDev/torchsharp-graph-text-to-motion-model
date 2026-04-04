using Microsoft.Extensions.Configuration;

public static class Inference
{
    public static void Run(IConfigurationRoot config)
    {
        Console.WriteLine("Running inference...");
    }
}
