using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Text2Motion.ClipModel;
using Text2Motion.DataPreprocessing;
using Text2Motion.Dataset;
using Text2Motion.TorchTrainer;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

using IHost host = Host.CreateDefaultBuilder(args)
    .ConfigureAppConfiguration((_, config) =>
    {
        config.AddJsonFile("training-settings.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("dataset-settings.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("preprocessing-config.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("ModelConfigs/BaselineMLPModelConfig.json", optional: false, reloadOnChange: true);
        config.AddJsonFile("ModelConfigs/StubModelConfig.json", optional: false, reloadOnChange: true);
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
        services.AddSingleton<TrainingMetricsService>();
        services.AddMotionModel<BaselineMLPModel, BaselineMLPModelConfig>(configuration);
        services.AddSingleton<HumanML3DDataset>();
        services.AddSingleton<TextToMotionModelTrainer>();
        services.Configure<TrainingConfig>(
            configuration.GetSection("Model"));
        services.Configure<DatasetSettings>(
            configuration.GetSection("Dataset"));
        services.Configure<TrainingSettings>(
            configuration.GetSection("Training"));
        services.Configure<PreprocessingConfig>(configuration);
    })
    .Build();

var lifetime = host.Services.GetRequiredService<IHostApplicationLifetime>();
var token = lifetime.ApplicationStopping;

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
