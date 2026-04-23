using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using static TorchSharp.torch;

namespace Text2Motion.TorchTrainer;

public static class ModelRegistrationExtensions
{
    public static IServiceCollection AddMotionModel<TModel, TConfig>(
        this IServiceCollection services,
        IConfiguration configuration)
        where TModel : nn.Module<Tensor, Tensor>
        where TConfig : class, new()
    {
        services.Configure<TConfig>(configuration.GetSection(typeof(TConfig).Name));
        services.AddSingleton<TModel>();
        services.AddSingleton(sp => (nn.Module<Tensor, Tensor>)sp.GetRequiredService<TModel>());
        return services;
    }
}
