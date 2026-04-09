using System.Diagnostics;
using System.Text.Json;
using Importer;
using Integrations.TwelveData;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Integrations.Services;
using Polly;
using Polly.Extensions.Http;


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

// Mode: --offline reads only from local cache, --online (default) fetches+updates via API
var offlineMode = !args.Contains("--online");
Console.WriteLine($"Mode: {(offlineMode ? "OFFLINE (cache-only)" : "ONLINE (cache + API)")}");

Console.WriteLine($"TwelveData Uri: {twelveDataSettings.Uri}");
var sw = Stopwatch.StartNew();

var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddSingleton<IConfiguration>(configuration);

if (!offlineMode)
{
    var retryPolicy = HttpPolicyExtensions
        .HandleTransientHttpError()
        .WaitAndRetryAsync(3, attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt)));

    var cacheHandler = new HttpResponseCacheHandler(
        cacheDirectory: Path.Combine(AppContext.BaseDirectory, "Cache", "HttpResponses"),
        useCache: false);

    builder.Services.AddSingleton<HttpResponseCacheHandler>(cacheHandler);
    builder.Services.AddHttpClients(new Uri(twelveDataSettings.Uri));
    builder.Services.AddHttpClient("TwelveData")
        .AddHttpMessageHandler<HttpResponseCacheHandler>()
        .AddPolicyHandler(retryPolicy);
}

builder.Services.AddSingleton<IRepository<TimeSeriesCacheDocument>>(_ =>
    new JsonFileRepository<TimeSeriesCacheDocument>(
        Path.Combine(AppContext.BaseDirectory, "Cache", "TimeSeries"),
        CacheJsonContext.Default.TimeSeriesCacheDocument));

builder.Services.AddSingleton<IRepository<IndicatorCacheDocument>>(_ =>
    new JsonFileRepository<IndicatorCacheDocument>(
        Path.Combine(AppContext.BaseDirectory, "Cache", "Indicators"),
        CacheJsonContext.Default.IndicatorCacheDocument));

var host = builder.Build();

HttpClient? httpClient = offlineMode
    ? null
    : host.Services.GetRequiredService<IHttpClientFactory>().CreateClient("TwelveData");
var repository = host.Services.GetRequiredService<IRepository<TimeSeriesCacheDocument>>();
var indicatorRepository = host.Services.GetRequiredService<IRepository<IndicatorCacheDocument>>();

var unavailableSymbolsPath = Path.Combine(AppContext.BaseDirectory, "Cache", "unavailable_symbols.json");
var unavailableSymbols = File.Exists(unavailableSymbolsPath)
    ? new HashSet<string>(JsonSerializer.Deserialize<List<string>>(File.ReadAllText(unavailableSymbolsPath)) ?? [], StringComparer.OrdinalIgnoreCase)
    : new HashSet<string>(StringComparer.OrdinalIgnoreCase);

var jsonOptions = new JsonSerializerOptions { WriteIndented = true };

void MarkUnavailable(string symbol, string reason)
{
    Console.WriteLine($"  {symbol}: not available — {reason}");
    unavailableSymbols.Add(symbol);
    File.WriteAllText(unavailableSymbolsPath, JsonSerializer.Serialize(unavailableSymbols.Order().ToList(), jsonOptions));
}

// Note: "1month" is excluded — variable month lengths cause cache timestamp misalignment.
string[] intervals = ["15min", "30min", "1h", "4h", "1day", "1week"];

string[] symbols =
[
    // Precious metals
    "XAU/USD", "XAG/USD", "XPT/USD", "XPD/USD",
    // Forex
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "USD/CHF", "USD/CNH", "USD/CNY",
    // ETFs
    // Removed (short history): "XMTH", "SUOD", "ZGLD", "WORLD"
    "SHY", "IEF", "TLT", "SPY", "QQQ", "EEM", "ZSL", "DGZ", "NDAQ", "IYY",
    // Oil ETFs
    // Removed (short history on 1day/1week, binding training start): "3SOI"
    // Removed (UK-listed ETF, no intraday data before 2022, causes 249 fully-NaN cols in early folds): "LOIL"
    // Commodities
    // Removed (short history): "CL1", "C_1", "NG/USD"
    // Removed (US-hours-only futures, 73-86% NaN on 4h): "HG1"
    "WTI/USD",
    // Indices
    // Removed (started Aug 2022, was binding training start to 3.5 years): "SVIX"
    "VIXY", "UDN", "UUP"
];

TwelveDataEndpoint[] volumeEndpoints =
[
    TwelveDataEndpoint.Obv,
    TwelveDataEndpoint.Ad,
    TwelveDataEndpoint.Adosc,
    TwelveDataEndpoint.Rvol,
];

TwelveDataParam MakeSeriesParam(string sym, string ivl, DateTime start, DateTime end) => new(
    httpClient: httpClient!,
    repository: repository,
    apiKey: twelveDataSettings.ApiKey!,
    symbol: sym,
    startDate: start,
    endDate: end,
    format: TwelveDataFormat.Json,
    interval: ivl,
    outputSize: 5000);

TwelveDataParam MakeIndicatorParam(string sym, TwelveDataEndpoint ep, string ivl, DateTime start, DateTime end) => new(
    httpClient: httpClient!,
    repository: repository,
    apiKey: twelveDataSettings.ApiKey!,
    symbol: sym,
    startDate: start,
    endDate: end,
    format: TwelveDataFormat.Json,
    endpoint: ep,
    interval: ivl,
    outputSize: 5000,
    indicatorRepository: indicatorRepository);

var fetchStart = new DateTime(1990, 1, 1, 0, 0, 0);
var fetchEnd = DateTime.UtcNow.Date;

var datasetRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Dataset"));

foreach (var interval in intervals)
{
    Console.WriteLine($"\n\n=== INTERVAL: {interval} ===");

    // ── Pass 1: fetch + trim all time series ──────────────────────────────────

    Console.WriteLine("\n--- Pass 1: fetching time series ---");

    var trimmedBySymbol = new Dictionary<string, Dictionary<DateTime, TimeSeriesValue>>();

    foreach (var symbol in symbols)
    {
        if (unavailableSymbols.Contains(symbol))
        {
            Console.WriteLine($"  {symbol}: skipping (not available on current plan)");
            continue;
        }

        Dictionary<DateTime, TimeSeriesValue> candles;
        try
        {
            if (offlineMode)
            {
                var doc = await repository.GetAsync(symbol, interval);
                candles = doc?.Intervals.TryGetValue(interval, out var bucket) == true
                    ? bucket.ToDictionary(
                        kvp => DateTime.ParseExact(kvp.Key, "yyyy-MM-dd HH:mm:ss", System.Globalization.CultureInfo.InvariantCulture),
                        kvp => kvp.Value)
                    : new Dictionary<DateTime, TimeSeriesValue>();
            }
            else
            {
                candles = await TwelveDataSeries.GetSeries(MakeSeriesParam(symbol, interval, fetchStart, fetchEnd));
            }
        }
        catch (TwelveDataSymbolUnavailableException ex)
        {
            MarkUnavailable(symbol, ex.Message);
            continue;
        }

        var trimmed = candles.TrimLeadingAndTrailingFilledCandles();

        if (trimmed.Count == 0)
        {
            Console.WriteLine($"  {symbol}: no data");
            continue;
        }

        trimmedBySymbol[symbol] = trimmed;
        Console.WriteLine($"  {symbol}: {trimmed.Count} candles  [{trimmed.Keys.Min():yyyy-MM-dd} → {trimmed.Keys.Max():yyyy-MM-dd}]");
    }

    if (trimmedBySymbol.Count == 0)
    {
        Console.WriteLine($"  No data for interval {interval}.");
        continue;
    }

    // ── Common start: latest first non-filled candle across all datasets ──────

    var commonStart = trimmedBySymbol.Values.Max(c => c.Keys.Min());
    Console.WriteLine($"\nCommon start: {commonStart:yyyy-MM-dd}  ({trimmedBySymbol.Count}/{symbols.Length} symbols with data)");

    // ── Write per-symbol start dates to report (append each interval) ─────────
    var commonStartReportPath = Path.Combine(datasetRoot, "common_start_report.txt");
    var symbolStartLines = trimmedBySymbol
        .Select(kvp => (symbol: kvp.Key, start: kvp.Value.Keys.Min()))
        .OrderByDescending(x => x.start)
        .Select(x => $"  {x.start:yyyy-MM-dd}  {x.symbol}");
    var isFirstInterval = !File.Exists(commonStartReportPath) || interval == intervals[0];
    if (isFirstInterval) File.WriteAllText(commonStartReportPath, "");
    File.AppendAllText(commonStartReportPath,
        $"\n[{interval}]  common start: {commonStart:yyyy-MM-dd}\n" +
        string.Join("\n", symbolStartLines) + "\n");

    // RVOL needs a warm-up period of at least 20 bars before commonStart.
    // For 1week, 20 bars = 140 days; for all shorter intervals 20 calendar days covers 20+ bars.
    var rvolLookback = interval == "1week" ? TimeSpan.FromDays(140) : TimeSpan.FromDays(20);
    var rvolStart = commonStart - rvolLookback;

    // ── Pass 2: filter to common start, compute indicators ───────────────────

    Console.WriteLine("\n--- Pass 2: indicators from common start ---");

    foreach (var (symbol, trimmed) in trimmedBySymbol)
    {
        var candles = trimmed
            .Where(kvp => kvp.Key >= commonStart)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        Console.WriteLine($"\n=== {symbol} ({candles.Count} candles) ===");

        // ── Volume indicators (API-computed, cached) ──────────────────────────

        foreach (var ep in volumeEndpoints)
        {
            var epPath = ep switch
            {
                TwelveDataEndpoint.Ad    => "ad",
                TwelveDataEndpoint.Adosc => "adosc",
                TwelveDataEndpoint.Obv   => "obv",
                TwelveDataEndpoint.Rvol  => "rvol",
                _                        => ep.ToString().ToLowerInvariant()
            };

            if (offlineMode)
            {
                var bucketKey = $"{epPath}/{interval}";
                var doc = await indicatorRepository.GetAsync(symbol, bucketKey);
                var count = doc?.Data.TryGetValue(bucketKey, out var bucket) == true ? bucket.Count : 0;
                Console.WriteLine($"  {ep,-8}: {count,8} (cached)");
            }
            else
            {
                var indicatorStart = ep == TwelveDataEndpoint.Rvol ? rvolStart : commonStart;
                var rawIndicator = await TwelveDataIndicator.GetIndicator(MakeIndicatorParam(symbol, ep, interval, indicatorStart, fetchEnd));
                var indicator = rawIndicator.TrimLeadingAndTrailingFilledIndicators();
                Console.WriteLine($"  {ep,-8}: {indicator.Count,8}");
            }
        }
    }
}

Console.WriteLine($"\nCompleted in {sw.Elapsed:hh\\:mm\\:ss}.");

public sealed record TwelveDataSettings
{
    public string? ApiKey { get; init; }
    public string? Uri { get; init; }
}
