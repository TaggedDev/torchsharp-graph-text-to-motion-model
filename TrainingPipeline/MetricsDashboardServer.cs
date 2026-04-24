using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace Text2Motion.TorchTrainer;

internal static class MetricsDashboardServer
{
    public static Task StartAsync(string outputRootPath, int port)
    {
        return Task.Run(async () =>
        {
            var builder = WebApplication.CreateSlimBuilder();
            builder.Logging.SetMinimumLevel(LogLevel.Warning);
            var app = builder.Build();
            app.Urls.Add($"http://localhost:{port}");

            app.MapGet("/", () => Results.Content(DashboardHtml, "text/html"));

            app.MapGet("/api/metrics", () =>
            {
                try
                {
                    var path = FindLatestMetricsPath(outputRootPath);
                    if (path is null || !File.Exists(path))
                        return Results.Content("{}", "application/json");
                    return Results.Content(File.ReadAllText(path), "application/json");
                }
                catch
                {
                    return Results.Content("{}", "application/json");
                }
            });

            await app.RunAsync();
        });
    }

    private static string? FindLatestMetricsPath(string outputRootPath)
    {
        if (!Directory.Exists(outputRootPath)) return null;
        var latest = Directory.GetDirectories(outputRootPath, "Run-*")
            .OrderByDescending(x => x)
            .FirstOrDefault();
        return latest is null ? null : Path.Combine(latest, "results", "metrics.json");
    }

    private const string DashboardHtml = """
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Training Dashboard</title>
          <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
          <style>
            body { background: #1e1e1e; color: #ccc; font-family: sans-serif; margin: 0; padding: 16px; }
            h2 { margin: 0 0 12px; }
            #status { font-size: 12px; color: #888; margin-bottom: 8px; }
          </style>
        </head>
        <body>
          <h2>Training Loss</h2>
          <div id="status">Waiting for data...</div>
          <div id="chart"></div>
          <script>
            const layout = {
              paper_bgcolor: '#1e1e1e', plot_bgcolor: '#2a2a2a',
              font: { color: '#ccc' },
              xaxis: { title: 'Epoch', gridcolor: '#444' },
              yaxis: { title: 'Loss', gridcolor: '#444' },
              legend: { orientation: 'h', y: -0.15 },
              margin: { t: 20, r: 20, b: 60, l: 60 }
            };
            const config = { responsive: true };
            let initialized = false;

            async function poll() {
              try {
                const r = await fetch('/api/metrics');
                const d = await r.json();
                const epochs = d.Epochs ?? [];
                const train = d.TrainLoss ?? [];
                const val = d.ValidationLoss ?? [];
                const test = d.TestLoss ?? [];

                if (train.length === 0) { document.getElementById('status').textContent = 'Waiting for data...'; return; }

                const xs = epochs.length > 0 ? epochs : train.map((_, i) => i + 1);

                const traces = [
                  { x: xs, y: train, name: 'Train Loss', mode: 'lines+markers', line: { color: '#4fc3f7' } },
                  { x: xs, y: val,   name: 'Val Loss',   mode: 'lines+markers', line: { color: '#ef9a9a' } }
                ];

                if (test.length > 0) {
                  traces.push({ x: xs, y: Array(xs.length).fill(test[0]), name: `Test Loss (${test[0].toFixed(4)})`, mode: 'lines', line: { color: '#a5d6a7', dash: 'dash' } });
                }

                if (!initialized) { Plotly.newPlot('chart', traces, layout, config); initialized = true; }
                else { Plotly.react('chart', traces, layout); }

                document.getElementById('status').textContent =
                  `Epoch ${xs.at(-1)} | train: ${train.at(-1).toFixed(6)} | val: ${val.at(-1).toFixed(6)}` +
                  (test.length ? ` | test: ${test[0].toFixed(6)}` : '');
              } catch(e) {
                document.getElementById('status').textContent = 'Error polling metrics.';
              }
            }

            poll();
            setInterval(poll, 2000);
          </script>
        </body>
        </html>
        """;
}
