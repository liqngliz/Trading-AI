using Integrations.TwelveData;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Integrations.Services;



var appsettings = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT")) ? "appsettings.dev.json": "appsettings.json";

var configBuilder = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile(appsettings)
    .AddEnvironmentVariables();

var configuration = configBuilder.Build();

var twelveDataSettings = configuration.GetSection("AppSettings:TwelveData").Get<TwelveDataSettings>();

Console.WriteLine($"TwelveData Uri: {twelveDataSettings?.Uri}");

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddHttpClients(new Uri("https://api.twelvedata.com/"));

var host = builder.Build();

var httpClientFactory = host.Services.GetRequiredService<IHttpClientFactory>();
var httpClient = httpClientFactory.CreateClient("TwelveData");

var param = new TwelveTimeSeriesParam(
    httpClient: httpClient,
    apiKey: twelveDataSettings?.ApiKey ?? throw new InvalidOperationException("TwelveData API key is required in configuration."),
    symbol: "XAU/USD",
    startDate: new DateTime(2022, 10, 1),
    endDate: new DateTime(2022, 11, 1),
    format: TwelveDataFormat.Json);

var candles = await param.GetSeries();

Console.WriteLine($"Candles fetched: {candles.Count}");
public record TwelveDataSettings
{
    public string? ApiKey { get; init; }
    public string? Uri { get; init; }
}