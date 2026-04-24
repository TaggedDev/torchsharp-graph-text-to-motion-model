using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Text2Motion.ClipModel;
using Text2Motion.DataPreprocessing;
using Text2Motion.Dataset;
using Text2Motion.TorchTrainer;

using IHost host = Host.CreateDefaultBuilder(args)
    .ConfigureAppConfiguration((_, config) =>
    {
        config.AddJsonFile("training-settings.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("dataset-settings.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("preprocessing-config.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("configs/BaselineMLPModelConfig.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("configs/StubModelConfig.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("configs/GcnSpatialTemporalConfig.json", optional: false, reloadOnChange: true);
        config.AddEnvironmentVariables(prefix: "AI_");
    })
    .ConfigureServices((context, services) =>
    {
        var configuration = context.Configuration;

        services.AddSingleton(configuration);
        services.AddSingleton<TrainingPipeline>();
        services.AddSingleton<ClipModelOnnxInference>();
        services.AddSingleton<DataPreprocessor>();
        services.AddSingleton<ModelCheckpointService>();
        services.AddMotionModel<GcnSpatialTemporalModel, GcnSpatialTemporalConfig>(configuration);
        services.AddSingleton<HumanML3DDataset>();
        services.AddSingleton<TextToMotionModelTrainer>();
        services.Configure<DatasetSettings>(
            configuration.GetSection("Dataset"));
        services.Configure<TrainingSettings>(
            configuration.GetSection("Training"));
        services.Configure<PreprocessingConfig>(configuration);
    })
    .Build();

var lifetime = host.Services.GetRequiredService<IHostApplicationLifetime>();
var token = lifetime.ApplicationStopping;

var trainingSettings = host.Services.GetRequiredService<IOptions<TrainingSettings>>().Value;
string outputRootPath = string.IsNullOrWhiteSpace(trainingSettings.OutputRootPath)
    ? Path.Combine(AppContext.BaseDirectory, "Weights", "Text2Motion")
    : Path.GetFullPath(trainingSettings.OutputRootPath);
_ = MetricsDashboardServer.StartAsync(outputRootPath, port: 5000, token);
Console.WriteLine("Dashboard: http://localhost:5000");

try
{
    using var scope = host.Services.CreateScope();
    var provider = scope.ServiceProvider;
    var trainingPipeline = provider.GetRequiredService<TrainingPipeline>();
    await trainingPipeline.ExecuteAsync(token);
}
catch (OperationCanceledException)
{
    Console.WriteLine("Graceful shutdown (cancellation requested).");
}
catch (Exception ex)
{
    Console.WriteLine($"Fatal error: {ex}");
}
