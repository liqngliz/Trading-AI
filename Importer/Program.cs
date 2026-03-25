using System.Diagnostics;
using System.Text.Json;
using Importer;
using Indicators;
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
string[] intervals = ["15min", "30min", "1h", "2h", "4h", "8h", "1day", "1week"];

// Transformer data: symbol → interval → sorted candles / indicator values
var allCandles       = new Dictionary<string, Dictionary<string, List<TimeSeriesValue>>>(StringComparer.OrdinalIgnoreCase);
var allIndicators    = new Dictionary<string, Dictionary<string, List<IndicatorValue>>>(StringComparer.OrdinalIgnoreCase);
var commonStartByTf  = new Dictionary<string, DateTime>(StringComparer.Ordinal);

var fetchStart = new DateTime(1990, 1, 1, 0, 0, 0);
var fetchEnd = DateTime.UtcNow.Date;

string[] symbols =
[
    // Precious metals
    "XAU/USD", "XAG/USD", "XPT/USD", "XPD/USD",
    // Forex
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "USD/CNH",
    // ETFs
    "XMTH", "SUOD", "SHY", "IEF", "TLT", "SPY", "EEM", "ZGLD","ZSL", "DGZ", "NDAQ", "QQQ", "IYY", "WORLD",
    // Oil ETFs
    "3SOI", "LOIL",
    // Commodities
    "WTI/USD", "CL1", "NG/USD", "HG1", "C_1", "C_1",
    // Indices
    "VIXY", "SVIX", "UDN", "UUP"
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

        if (!allCandles.TryGetValue(symbol, out var symCandlesByTf))
            allCandles[symbol] = symCandlesByTf = new Dictionary<string, List<TimeSeriesValue>>(StringComparer.Ordinal);
        symCandlesByTf[interval] = trimmed.Values.OrderBy(v => v.Datetime).ToList();
    }

    if (trimmedBySymbol.Count == 0)
    {
        Console.WriteLine($"  No data for interval {interval}.");
        continue;
    }

    // ── Common start: latest first non-filled candle across all datasets ──────

    var commonStart = trimmedBySymbol.Values.Max(c => c.Keys.Min());
    commonStartByTf[interval] = commonStart;
    Console.WriteLine($"\nCommon start: {commonStart:yyyy-MM-dd}  ({trimmedBySymbol.Count}/{symbols.Length} symbols with data)");

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

        // ── Price indicators ──────────────────────────────────────────────────

        Console.WriteLine($"  LogReturn:        {PriceIndicators.LogReturn(candles).Count}");
        Console.WriteLine($"  SMA(20):          {PriceIndicators.Sma(candles, 20).Count}");
        Console.WriteLine($"  EMA(20):          {PriceIndicators.Ema(candles, 20).Count}");
        Console.WriteLine($"  RSI(14):          {PriceIndicators.Rsi(candles).Count}");
        Console.WriteLine($"  ROC(1):           {PriceIndicators.Roc(candles).Count}");
        Console.WriteLine($"  StdDev(20):       {PriceIndicators.RollingStdDev(candles, 20).Count}");
        Console.WriteLine($"  ATR(14):          {PriceIndicators.Atr(candles).Count}");
        Console.WriteLine($"  Bollinger(20):    {PriceIndicators.BollingerBands(candles).Count}");
        Console.WriteLine($"  CCI(20):          {PriceIndicators.Cci(candles).Count}");
        Console.WriteLine($"  WilliamsR(14):    {PriceIndicators.WilliamsR(candles).Count}");
        Console.WriteLine($"  Stochastic:       {PriceIndicators.Stochastic(candles).Count}");
        Console.WriteLine($"  MACD:             {PriceIndicators.Macd(candles).Count}");
        Console.WriteLine($"  CandleRange:      {PriceIndicators.CandleRange(candles).Count}");
        Console.WriteLine($"  BodySize:         {PriceIndicators.BodySize(candles).Count}");
        Console.WriteLine($"  UpperWick:        {PriceIndicators.UpperWick(candles).Count}");
        Console.WriteLine($"  LowerWick:        {PriceIndicators.LowerWick(candles).Count}");
        Console.WriteLine($"  DistHighN(20):    {PriceIndicators.DistanceFromHighN(candles, 20).Count}");
        Console.WriteLine($"  DistLowN(20):     {PriceIndicators.DistanceFromLowN(candles, 20).Count}");

        // ── Volume indicators (API-computed, cached, trimmed) ─────────────────

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

            Dictionary<DateTime, IndicatorValue> rawIndicator;
            if (offlineMode)
            {
                var bucketKey = $"{epPath}/{interval}";
                var doc = await indicatorRepository.GetAsync(symbol, bucketKey);
                rawIndicator = doc?.Data.TryGetValue(bucketKey, out var bucket) == true
                    ? bucket.ToDictionary(
                        kvp => DateTime.ParseExact(kvp.Key, "yyyy-MM-dd HH:mm:ss", System.Globalization.CultureInfo.InvariantCulture),
                        kvp => kvp.Value)
                    : [];
            }
            else
            {
                var indicatorStart = ep == TwelveDataEndpoint.Rvol ? rvolStart : commonStart;
                rawIndicator = await TwelveDataIndicator.GetIndicator(MakeIndicatorParam(symbol, ep, interval, indicatorStart, fetchEnd));
            }

            var indicator = rawIndicator.TrimLeadingAndTrailingFilledIndicators();
            Console.WriteLine($"  {ep,-8}:         {indicator.Count}");

            if (!allIndicators.TryGetValue(symbol, out var symIndsByKey))
                allIndicators[symbol] = symIndsByKey = new Dictionary<string, List<IndicatorValue>>(StringComparer.Ordinal);
            symIndsByKey[$"{epPath}/{interval}"] = indicator.Values.OrderBy(v => v.Datetime).ToList();
        }
    }
}

// ── Build XAU/USD 4h ML dataset ───────────────────────────────────────────────

Console.WriteLine("\n\n=== TRANSFORMER: building XAU/USD 4h dataset ===");

var trainingStart = commonStartByTf.TryGetValue("4h", out var cs4h) ? cs4h : (DateTime?)null;

var cfg = new DatasetConfig
{
    TargetSymbol       = "XAU/USD",
    TargetHorizons     = [("4h", 1), ("8h", 1), ("1day", 1), ("1week", 1)],
    TrainingStartDate  = trainingStart,
    OutputDirectory    = Path.Combine(AppContext.BaseDirectory, "Dataset")
};

var (rows, columns, columnStats) = Transformer.BuildFeatureMatrix(cfg, allCandles, allIndicators);

if (rows.Count > 0)
{
    var csvPath      = Path.Combine(cfg.OutputDirectory, "xauusd_4h_dataset.csv");
    var bucketedPath = Path.Combine(cfg.OutputDirectory, "xauusd_4h_dataset_bucketed.csv");
    var reportPath   = Path.Combine(cfg.OutputDirectory, "xauusd_4h_report.txt");

    var nullReportPath    = Path.Combine(cfg.OutputDirectory, "xauusd_4h_null_report.json");
    var prunedColsPath    = Path.Combine(cfg.OutputDirectory, "xauusd_4h_pruned_columns.txt");

    (rows, columns) = Transformer.PruneSparseCols(rows, columns, prunedColsPath);

    Transformer.ExportToCsv(rows, columns, csvPath);
    Transformer.BucketTargets(csvPath, bucketedPath);
    Transformer.CreateWalkForwardSplits(rows, columns, cfg);
    Transformer.GenerateDatasetReport(rows, columns, reportPath);
    Transformer.GenerateNullReport(rows, columns, nullReportPath);
}

Console.WriteLine($"\nCompleted in {sw.Elapsed:hh\\:mm\\:ss}.");

public sealed record TwelveDataSettings
{
    public string? ApiKey { get; init; }
    public string? Uri { get; init; }
}
