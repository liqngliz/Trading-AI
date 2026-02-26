using Microsoft.Extensions.DependencyInjection;

namespace Integrations.Services;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddHttpClients(
        this IServiceCollection services,
        Uri twelveDataBaseUri)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(twelveDataBaseUri);

        services.AddHttpClient("TwelveData", client =>
        {
            client.BaseAddress = twelveDataBaseUri;
            client.Timeout = TimeSpan.FromSeconds(60);
        });

        return services;
    }
}