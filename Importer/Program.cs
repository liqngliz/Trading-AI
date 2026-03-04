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

// Register JSON repositories in DI — separate subdirectories to avoid filename collisions
builder.Services.AddSingleton<IRepository<TimeSeriesCacheDocument>>(_ =>
    new JsonFileRepository<TimeSeriesCacheDocument>(
        Path.Combine(AppContext.BaseDirectory, "Cache", "TimeSeries")));

builder.Services.AddSingleton<IRepository<IndicatorCacheDocument>>(_ =>
    new JsonFileRepository<IndicatorCacheDocument>(
        Path.Combine(AppContext.BaseDirectory, "Cache", "Indicators")));

var host = builder.Build();

var httpClientFactory = host.Services.GetRequiredService<IHttpClientFactory>();
var httpClient = httpClientFactory.CreateClient("TwelveData");

var repository = host.Services.GetRequiredService<IRepository<TimeSeriesCacheDocument>>();
var indicatorRepository = host.Services.GetRequiredService<IRepository<IndicatorCacheDocument>>();

const string symbol = "XAU/USD";
const string interval = "4h";
var startDate = new DateTime(1992, 10, 1, 0, 0, 0);
var endDate = new DateTime(2025, 11, 1, 0, 0, 0);

var param = new TwelveDataParam(
    httpClient: httpClient,
    repository: repository,
    apiKey: twelveDataSettings.ApiKey,
    symbol: symbol,
    startDate: startDate,
    endDate: endDate,
    format: TwelveDataFormat.Json,
    interval: interval,
    outputSize: 5000);

// ── Time series ───────────────────────────────────────────────────────────────

var candles = await TwelveDataSeries.GetSeries(param);

Console.WriteLine($"Candles fetched: {candles.Count}");
var trimmedLeadingCandles = candles.TrimLeadingFilledCandles();
Console.WriteLine($"Trimmed leading candles count: {trimmedLeadingCandles.Count}");
Console.WriteLine($"Trimmed leading candles start: {trimmedLeadingCandles.Keys.FirstOrDefault()}");
var trimmedTrailingCandles = candles.TrimTrailingFilledCandles();
Console.WriteLine($"Trimmed trailing candles count: {trimmedTrailingCandles.Count}");
Console.WriteLine($"Trimmed trailing candles end: {trimmedTrailingCandles.Keys.LastOrDefault()}");
var trimmedLeadingAndTrailingCandles = candles.TrimLeadingAndTrailingFilledCandles();
Console.WriteLine($"Trimmed leading and trailing candles count: {trimmedLeadingAndTrailingCandles.Count}");
Console.WriteLine($"Trimmed leading and trailing candles start: {trimmedLeadingAndTrailingCandles.Keys.FirstOrDefault()}");
Console.WriteLine($"Trimmed leading and trailing candles end: {trimmedLeadingAndTrailingCandles.Keys.LastOrDefault()}");


// ── Volume indicators ─────────────────────────────────────────────────────────

TwelveDataParam IndicatorParam(TwelveDataEndpoint endpoint) => new(
    httpClient: httpClient,
    repository: repository,
    apiKey: twelveDataSettings.ApiKey,
    symbol: symbol,
    startDate: startDate,
    endDate: endDate,
    format: TwelveDataFormat.Json,
    endpoint: endpoint,
    interval: interval,
    outputSize: 5000,
    indicatorRepository: indicatorRepository);
/*
var obv = (await TwelveDataIndicator.GetIndicator(IndicatorParam(TwelveDataEndpoint.Obv))).TrimLeadingAndTrailingFilledIndicators();
Console.WriteLine($"OBV count: {obv.Count}, start: {obv.Keys.FirstOrDefault()}, end: {obv.Keys.LastOrDefault()}");

var ad = (await TwelveDataIndicator.GetIndicator(IndicatorParam(TwelveDataEndpoint.Ad))).TrimLeadingAndTrailingFilledIndicators();
Console.WriteLine($"AD  count: {ad.Count}, start: {ad.Keys.FirstOrDefault()}, end: {ad.Keys.LastOrDefault()}");

var adosc = (await TwelveDataIndicator.GetIndicator(IndicatorParam(TwelveDataEndpoint.Adosc))).TrimLeadingAndTrailingFilledIndicators();
Console.WriteLine($"ADOSC count: {adosc.Count}, start: {adosc.Keys.FirstOrDefault()}, end: {adosc.Keys.LastOrDefault()}");
*/

var rvol = (await TwelveDataIndicator.GetIndicator(IndicatorParam(TwelveDataEndpoint.Rvol))).TrimLeadingAndTrailingFilledIndicators();
Console.WriteLine($"RVOL  count: {rvol.Count}, start: {rvol.Keys.FirstOrDefault()}, end: {rvol.Keys.LastOrDefault()}");
public sealed record TwelveDataSettings
{
    public string? ApiKey { get; init; }
    public string? Uri { get; init; }
}