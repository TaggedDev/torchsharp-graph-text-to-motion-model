// Quick CUDA check script
// Run: dotnet script check_cuda.cs

using TorchSharp;
using static TorchSharp.torch;

Console.WriteLine("=== CUDA DIAGNOSTICS ===");
Console.WriteLine($"CUDA Available: {cuda.is_available()}");

if (cuda.is_available())
{
    Console.WriteLine($"Device Count: {cuda.device_count()}");
    Console.WriteLine("GPU Training ENABLED ✓");
}
else
{
    Console.WriteLine("WARNING: CUDA not available - training will use CPU");
    Console.WriteLine("\nChecks:");
    Console.WriteLine("1. Is NVIDIA GPU present?");
    Console.WriteLine("2. Is CUDA 12.8 installed?");
    Console.WriteLine("3. Is cuDNN installed?");
    Console.WriteLine("4. Set environment: CUDA_VISIBLE_DEVICES=0");
}
