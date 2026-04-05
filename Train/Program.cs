using Microsoft.Extensions.Configuration;
using TorchSharp;

public static class Train
{
    public static void Run(IConfigurationRoot config)
    {
        torch.InitializeDeviceType(DeviceType.CUDA);
        torch.manual_seed(42);

        Console.WriteLine($"CUDA available: {torch.cuda.is_available()}");
        Console.WriteLine($"Device count: {torch.cuda.device_count()}");

        var trainer = new Training.Trainer(config);
        trainer.Run();
    }
}
