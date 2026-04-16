using Text2Motion.DataPreprocessing;
using Text2Motion.TorchTrainer;

internal class TrainingPipeline(DataPreprocessor preprocessor, TextToMotionModelTrainer trainer)
{
    public async Task ExecuteAsync(CancellationToken token)
    {
        Console.WriteLine("Starting preprocessing...");
        await preprocessor.RunAsync(token);

        Console.WriteLine("Starting training...");
        await trainer.TrainAsync(token);

        Console.WriteLine("Pipeline completed.");
    }
}