using Microsoft.Extensions.Configuration;

public static class Train
{
    public static void Run(IConfigurationRoot config)
    {
        Console.WriteLine("Running training...");
    }
}
