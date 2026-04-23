using System.Diagnostics;
using TorchSharp;

namespace Text2Motion.TorchTrainer;

public class PerformanceMonitor
{
    private readonly Dictionary<string, Stopwatch> _activeTimers = new();
    private readonly Dictionary<string, List<long>> _completedTimings = new();

    public void StartTimer(string label)
    {
        if (!_completedTimings.ContainsKey(label))
            _completedTimings[label] = new();

        var sw = Stopwatch.StartNew();
        _activeTimers[label] = sw;
    }

    public long EndTimer(string label)
    {
        if (!_activeTimers.TryGetValue(label, out var sw))
            return 0;

        sw.Stop();
        var elapsed = sw.ElapsedMilliseconds;

        if (_completedTimings.TryGetValue(label, out var list))
            list.Add(elapsed);

        _activeTimers.Remove(label);
        return elapsed;
    }

    public void PrintSummary()
    {
        Console.WriteLine("\n=== PERFORMANCE SUMMARY ===");

        foreach (var (label, timings) in _completedTimings.OrderBy(x => x.Key))
        {
            if (timings.Count == 0)
                continue;

            var total = timings.Sum();
            var avg = timings.Average();
            var min = timings.Min();
            var max = timings.Max();

            Console.WriteLine($"{label}:");
            Console.WriteLine($"  Count: {timings.Count}, Total: {total}ms, Avg: {avg:F2}ms, Min: {min}ms, Max: {max}ms");
        }

        if (TorchSharp.torch.cuda.is_available())
        {
            Console.WriteLine($"\nGPU Training completed on CUDA device");
        }
        else
        {
            Console.WriteLine($"\nWARNING: Training completed on CPU (CUDA not available)");
            Console.WriteLine("Check device configuration and CUDA installation");
        }
    }
}
