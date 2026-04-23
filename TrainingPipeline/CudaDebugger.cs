using TorchSharp;
using static TorchSharp.torch;

namespace Text2Motion.TorchTrainer;

public static class CudaDebugger
{
    public static void PrintDeviceInfo(Device device)
    {
        Console.WriteLine($"\n=== DEVICE INFO ===");
        Console.WriteLine($"Device: {device}");
        Console.WriteLine($"CUDA Available: {cuda.is_available()}");
    }

    public static void PrintTensorInfo(string name, Tensor tensor)
    {
        Console.WriteLine($"[TENSOR] {name}");
        Console.WriteLine($"  Shape: {string.Join("x", tensor.shape)}");
        Console.WriteLine($"  Device: {tensor.device}");
        Console.WriteLine($"  DType: {tensor.dtype}");
        Console.WriteLine($"  Requires Grad: {tensor.requires_grad}");

        var numel = tensor.numel();
        var bytes = numel * sizeof(float);
        Console.WriteLine($"  Elements: {numel}, Approx Memory: {bytes / 1024 / 1024}MB");
    }

    public static void SynchronizeGpu()
    {
        if (cuda.is_available())
        {
            try
            {
                cuda.synchronize();
            }
            catch { }
        }
    }
}
