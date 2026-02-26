using Integrations.TwelveData;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Integrations.Services;


var appsettings = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT"))
    ? "appsettings.dev.json"
    : "appsettings.json";

var configuration = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile(appsettings, optional: false, reloadOnChange: false)
    .AddEnvironmentVariables()
    .Build();

var twelveDataSettings = configuration
    .GetSection("AppSettings:TwelveData")
    .Get<TwelveDataSettings>()
    ?? throw new InvalidOperationException("Missing configuration section: AppSettings:TwelveData");

if (string.IsNullOrWhiteSpace(twelveDataSettings.Uri))
    throw new InvalidOperationException("TwelveData Uri is required in configuration.");

if (string.IsNullOrWhiteSpace(twelveDataSettings.ApiKey))
    throw new InvalidOperationException("TwelveData API key is required in configuration.");

Console.WriteLine($"TwelveData Uri: {twelveDataSettings.Uri}");

var builder = Host.CreateApplicationBuilder(args);

// Register configuration if you want it available through DI
builder.Services.AddSingleton<IConfiguration>(configuration);

// Register named HttpClient from configuration
builder.Services.AddHttpClients(new Uri(twelveDataSettings.Uri));

// Register JSON repository in DI
builder.Services.AddSingleton<IRepository<TimeSeriesCacheDocument>>(_ =>
{
    var cacheDirectory = Path.Combine(AppContext.BaseDirectory, "Cache");
    Directory.CreateDirectory(cacheDirectory);

    return new JsonFileRepository<TimeSeriesCacheDocument>(cacheDirectory);
});

var host = builder.Build();

var httpClientFactory = host.Services.GetRequiredService<IHttpClientFactory>();
var httpClient = httpClientFactory.CreateClient("TwelveData");

var repository = host.Services.GetRequiredService<IRepository<TimeSeriesCacheDocument>>();

var param = new TwelveTimeSeriesParam(
    httpClient: httpClient,
    repository: repository,
    apiKey: twelveDataSettings.ApiKey,
    symbol: "XAU/USD",
    startDate: new DateTime(1992, 10, 1, 0, 0, 0),
    endDate: new DateTime(2025, 11, 1, 0, 0, 0),
    format: TwelveDataFormat.Json,
    interval: "4h",
    outputSize: 5000);

var candles = await param.GetSeries();

Console.WriteLine($"Candles fetched: {candles.Count}");

public sealed record TwelveDataSettings
{
    public string? ApiKey { get; init; }
    public string? Uri { get; init; }
}