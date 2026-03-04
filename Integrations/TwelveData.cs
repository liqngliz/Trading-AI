using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Text.Json;

namespace Integrations.TwelveData;

public enum TwelveDataEndpoint
{
    TimeSeries,
    Ad,
    Adosc,
    Obv,
    Rvol
}

public enum TwelveDataFormat
{
    Json,
    Csv
}

public readonly record struct TimeSeriesValue(
    DateTime Datetime,
    decimal Open,
    decimal High,
    decimal Low,
    decimal Close,
    bool IsFilled);

public readonly record struct IndicatorValue(
    DateTime Datetime,
    decimal Value,
    bool IsFilled);

public sealed record TimeSeriesCacheDocument
{
    public string Symbol { get; init; } = string.Empty;

    public Dictionary<string, SortedDictionary<string, TimeSeriesValue>> Intervals { get; init; } = new();
}

public sealed record IndicatorCacheDocument
{
    public string Symbol { get; init; } = string.Empty;

    // Outer key: "endpoint/interval" (e.g. "obv/4h"), inner key: storage key, value: IndicatorValue
    public Dictionary<string, SortedDictionary<string, IndicatorValue>> Data { get; init; } = new();
}

public sealed record TwelveDataFunctions<TValue>(
    Func<DateTime, DateTime, string, List<DateTime>> BuildExpectedTimestamps,
    Func<IReadOnlyList<DateTime>, IReadOnlyDictionary<string, TValue>, List<(DateTime Start, DateTime End)>> BuildMissingRanges,
    Func<DateTime, string> ToStorageKey,
    Func<string, DateTime> ParseStorageKey,
    Func<TwelveDataEndpoint, string> GetEndpointPath)
{
    public static TwelveDataFunctions<TValue> Default { get; } = new(
        TwelveDataParamExtensions.BuildExpectedTimestamps,
        TwelveDataParamExtensions.BuildMissingRanges,
        TwelveDataParamExtensions.ToStorageKey,
        TwelveDataParamExtensions.ParseStorageKey,
        TwelveDataParamExtensions.GetEndpointPath);
}

public sealed record TwelveDataParam
{
    public HttpClient HttpClient { get; }
    public IRepository<TimeSeriesCacheDocument> Repository { get; }
    public string ApiKey { get; }
    public string Symbol { get; }
    public DateTime StartDate { get; }
    public DateTime EndDate { get; }
    public TwelveDataFormat Format { get; }
    public TwelveDataEndpoint Endpoint { get; }
    public string Interval { get; }
    public int OutputSize { get; }
    public IRepository<IndicatorCacheDocument>? IndicatorRepository { get; }
    public CancellationToken CancellationToken { get; }

    public TwelveDataParam(
        HttpClient httpClient,
        IRepository<TimeSeriesCacheDocument> repository,
        string apiKey,
        string symbol,
        DateTime startDate,
        DateTime endDate,
        TwelveDataFormat format,
        TwelveDataEndpoint endpoint = TwelveDataEndpoint.TimeSeries,
        string interval = "4h",
        int outputSize = 5000,
        IRepository<IndicatorCacheDocument>? indicatorRepository = null,
        CancellationToken cancellationToken = default)
    {
        if (httpClient is null) throw new ArgumentNullException(nameof(httpClient));
        if (httpClient.BaseAddress is null)
            throw new InvalidOperationException("HttpClient.BaseAddress must be set.");
        if (repository is null) throw new ArgumentNullException(nameof(repository));
        if (string.IsNullOrWhiteSpace(apiKey)) throw new ArgumentException("API key is required.", nameof(apiKey));
        if (string.IsNullOrWhiteSpace(symbol)) throw new ArgumentException("Symbol is required.", nameof(symbol));
        if (startDate > endDate) throw new ArgumentException("startDate must be <= endDate.", nameof(startDate));
        if (string.IsNullOrWhiteSpace(interval)) throw new ArgumentException("Interval is required.", nameof(interval));
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));

        HttpClient = httpClient;
        Repository = repository;
        ApiKey = apiKey;
        Symbol = symbol;
        StartDate = startDate;
        EndDate = endDate;
        Format = format;
        Endpoint = endpoint;
        Interval = interval;
        OutputSize = outputSize;
        IndicatorRepository = indicatorRepository;
        CancellationToken = cancellationToken;
    }
}

public static class TwelveDataParamExtensions
{
    internal static List<DateTime> BuildExpectedTimestamps(
        DateTime start,
        DateTime end,
        string interval)
    {
        var step = ParseInterval(interval);
        var result = new List<DateTime>();

        for (var current = start; current <= end; current = current.Add(step))
        {
            if(current.Add(step) >= end && current != end)
                break;
            result.Add(current);
        }

        return result;
    }

    internal static List<(DateTime Start, DateTime End)> BuildMissingRanges<TValue>(
        IReadOnlyList<DateTime> expectedTimestamps,
        IReadOnlyDictionary<string, TValue> intervalBucket)
    {
        var missing = new List<DateTime>();

        foreach (var ts in expectedTimestamps)
        {
            if (!intervalBucket.ContainsKey(ToStorageKey(ts)))
            {
                missing.Add(ts);
            }
        }

        if (missing.Count == 0)
            return new List<(DateTime, DateTime)>();

        var ranges = new List<(DateTime Start, DateTime End)>();
        var step = missing.Count > 1
            ? missing[1] - missing[0]
            : TimeSpan.Zero;

        var rangeStart = missing[0];
        var previous = missing[0];

        for (int i = 1; i < missing.Count; i++)
        {
            var current = missing[i];
            var currentGap = current - previous;

            if (step != TimeSpan.Zero && currentGap != step)
            {
                ranges.Add((rangeStart, previous));
                rangeStart = current;
            }

            previous = current;
        }

        ranges.Add((rangeStart, previous));
        return ranges;
    }

    internal static TimeSpan ParseInterval(string interval)
    {
        return interval switch
        {
            "1min" => TimeSpan.FromMinutes(1),
            "5min" => TimeSpan.FromMinutes(5),
            "15min" => TimeSpan.FromMinutes(15),
            "30min" => TimeSpan.FromMinutes(30),
            "45min" => TimeSpan.FromMinutes(45),
            "1h" => TimeSpan.FromHours(1),
            "2h" => TimeSpan.FromHours(2),
            "4h" => TimeSpan.FromHours(4),
            "5h" => TimeSpan.FromHours(5),
            "1day" => TimeSpan.FromDays(1),
            "1week" => TimeSpan.FromDays(7),
            "1month" => throw new InvalidOperationException(
                "Interval '1month' is not supported by this implementation because month length is variable."),
            _ => throw new InvalidOperationException(
                $"Unsupported interval '{interval}'. Supported values: " +
                "1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 5h, 1day, 1week.")
        };
    }

    internal static string ToStorageKey(DateTime value) =>
        value.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

    internal static DateTime ParseStorageKey(string value) =>
        DateTime.ParseExact(value, "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

    internal static bool IsNoDataResponse(JsonElement root)
    {
        if (!root.TryGetProperty("status", out var statusElement) ||
            !string.Equals(statusElement.GetString(), "error", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (!root.TryGetProperty("code", out var codeElement) || codeElement.ValueKind != JsonValueKind.Number)
            return false;

        if (codeElement.GetInt32() != 400)
            return false;

        if (!root.TryGetProperty("message", out var messageElement))
            return false;

        var message = messageElement.GetString();
        return message?.Contains("No data is available on the specified dates", StringComparison.OrdinalIgnoreCase) == true;
    }

    internal static decimal ParseDecimal(JsonElement item, string propertyName)
    {
        var s = item.GetProperty(propertyName).GetString();
        if (string.IsNullOrWhiteSpace(s))
            throw new InvalidOperationException($"Missing '{propertyName}'.");

        return decimal.Parse(s, NumberStyles.Number, CultureInfo.InvariantCulture);
    }

    internal static string GetEndpointPath(TwelveDataEndpoint endpoint) => endpoint switch
    {
        TwelveDataEndpoint.TimeSeries => "time_series",
        TwelveDataEndpoint.Ad => "ad",
        TwelveDataEndpoint.Adosc => "adosc",
        TwelveDataEndpoint.Obv => "obv",
        TwelveDataEndpoint.Rvol => "rvol",
        _ => throw new ArgumentOutOfRangeException(nameof(endpoint), endpoint, null)
    };

    internal static class QueryHelpers
    {
        public static string AddQueryString(string uri, IEnumerable<KeyValuePair<string?, string?>> queryString)
        {
            var hasQuery = uri.Contains('?', StringComparison.Ordinal);
            var sb = new StringBuilder(uri);

            foreach (var kv in queryString)
            {
                if (kv.Key is null) continue;

                sb.Append(hasQuery ? '&' : '?');
                hasQuery = true;

                sb.Append(Uri.EscapeDataString(kv.Key));
                sb.Append('=');
                sb.Append(Uri.EscapeDataString(kv.Value ?? string.Empty));
            }

            return sb.ToString();
        }
    }
}
