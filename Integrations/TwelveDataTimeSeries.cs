using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Text.Json;

namespace Integrations.TwelveData;

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
    decimal Close);

public sealed record TimeSeriesCacheDocument
{
    public string Symbol { get; init; } = string.Empty;

    public Dictionary<string, SortedDictionary<string, TimeSeriesValue>> Intervals { get; init; } = new();
}

public sealed record TwelveTimeSeriesParam
{
    public HttpClient HttpClient { get; }
    public IRepository<TimeSeriesCacheDocument> Repository { get; }
    public string ApiKey { get; }
    public string Symbol { get; }
    public DateTime StartDate { get; }
    public DateTime EndDate { get; }
    public TwelveDataFormat Format { get; }
    public string Interval { get; }
    public int OutputSize { get; }
    public CancellationToken CancellationToken { get; }

    public TwelveTimeSeriesParam(
        HttpClient httpClient,
        IRepository<TimeSeriesCacheDocument> repository,
        string apiKey,
        string symbol,
        DateTime startDate,
        DateTime endDate,
        TwelveDataFormat format,
        string interval = "4h",
        int outputSize = 5000,
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
        Interval = interval;
        OutputSize = outputSize;
        CancellationToken = cancellationToken;
    }
}

public static class TwelveTimeSeriesParamExtensions
{
    public static async Task<Dictionary<DateTime, TimeSeriesValue>> GetSeries(
        this TwelveTimeSeriesParam param)
    {
        ArgumentNullException.ThrowIfNull(param);

        var cacheKey = param.Symbol;
        var cache = await param.Repository
            .GetAsync(cacheKey, param.CancellationToken)
            .ConfigureAwait(false)
            ?? new TimeSeriesCacheDocument
            {
                Symbol = param.Symbol
            };

        if (!cache.Intervals.TryGetValue(param.Interval, out var intervalBucket))
        {
            intervalBucket = new SortedDictionary<string, TimeSeriesValue>(StringComparer.Ordinal);
            cache.Intervals[param.Interval] = intervalBucket;
        }

        var expectedTimestamps = BuildExpectedTimestamps(
            param.StartDate,
            param.EndDate,
            param.Interval);

        var missingRanges = BuildMissingRanges(expectedTimestamps, intervalBucket);

        foreach (var missingRange in missingRanges)
        {
            var partialParam = new TwelveTimeSeriesParam(
                httpClient: param.HttpClient,
                repository: param.Repository,
                apiKey: param.ApiKey,
                symbol: param.Symbol,
                startDate: missingRange.Start,
                endDate: missingRange.End,
                format: param.Format,
                interval: param.Interval,
                outputSize: param.OutputSize,
                cancellationToken: param.CancellationToken);

            var fetched = await GetSeriesFromApi(partialParam).ConfigureAwait(false);

            foreach (var kvp in fetched)
            {
                intervalBucket[ToStorageKey(kvp.Key)] = kvp.Value;
            }
        }

        if (missingRanges.Count > 0)
        {
            await param.Repository
                .SaveAsync(cacheKey, cache, param.CancellationToken)
                .ConfigureAwait(false);
        }

        return intervalBucket
            .Where(x =>
            {
                var dt = ParseStorageKey(x.Key);
                return dt >= param.StartDate && dt <= param.EndDate;
            })
            .ToDictionary(
                x => ParseStorageKey(x.Key),
                x => x.Value);
    }

    public static async Task<Dictionary<DateTime, TimeSeriesValue>> GetNextSeries(
        this TwelveTimeSeriesParam param,
        IReadOnlyDictionary<DateTime, TimeSeriesValue> currentBatch)
    {
        ArgumentNullException.ThrowIfNull(param);
        ArgumentNullException.ThrowIfNull(currentBatch);

        if (currentBatch.Count == 0)
            return new Dictionary<DateTime, TimeSeriesValue>();

        var oldest = currentBatch.Keys.Min();

        if (oldest <= param.StartDate)
            return new Dictionary<DateTime, TimeSeriesValue>();

        var nextParam = new TwelveTimeSeriesParam(
            httpClient: param.HttpClient,
            repository: param.Repository,
            apiKey: param.ApiKey,
            symbol: param.Symbol,
            startDate: param.StartDate,
            endDate: oldest.AddTicks(-1),
            format: param.Format,
            interval: param.Interval,
            outputSize: param.OutputSize,
            cancellationToken: param.CancellationToken);

        return await nextParam.GetSeries().ConfigureAwait(false);
    }

    public static async Task<Dictionary<DateTime, TimeSeriesValue>> GetAllSeries(
        this TwelveTimeSeriesParam param)
    {
        ArgumentNullException.ThrowIfNull(param);

        var all = new Dictionary<DateTime, TimeSeriesValue>();
        var batch = await param.GetSeries().ConfigureAwait(false);

        while (batch.Count > 0)
        {
            var previousOldest = batch.Keys.Min();

            foreach (var kvp in batch)
            {
                all[kvp.Key] = kvp.Value;
            }

            var next = await param.GetNextSeries(batch).ConfigureAwait(false);

            if (next.Count == 0)
                break;

            var nextOldest = next.Keys.Min();
            if (nextOldest >= previousOldest)
                break;

            batch = next;
        }

        return all;
    }

    private static async Task<Dictionary<DateTime, TimeSeriesValue>> GetSeriesFromApi(
        TwelveTimeSeriesParam param)
    {
        var formatString = param.Format switch
        {
            TwelveDataFormat.Json => "json",
            TwelveDataFormat.Csv => "csv",
            _ => throw new ArgumentOutOfRangeException(nameof(param.Format), param.Format, null)
        };

        var query = new List<KeyValuePair<string?, string?>>
        {
            new("apikey", param.ApiKey),
            new("interval", param.Interval),
            new("symbol", param.Symbol),
            new("start_date", param.StartDate.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture)),
            new("end_date", param.EndDate.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture)),
            new("format", formatString),
            new("outputsize", param.OutputSize.ToString(CultureInfo.InvariantCulture))
        };

        var requestUri = new Uri(QueryHelpers.AddQueryString("time_series", query), UriKind.Relative);

        using var response = await param.HttpClient
            .GetAsync(requestUri, param.CancellationToken)
            .ConfigureAwait(false);

        response.EnsureSuccessStatusCode();

        var content = await response.Content
            .ReadAsStringAsync(param.CancellationToken)
            .ConfigureAwait(false);

        return param.Format == TwelveDataFormat.Json
            ? ParseJson(content)
            : ParseCsv(content);
    }

    private static List<DateTime> BuildExpectedTimestamps(
        DateTime start,
        DateTime end,
        string interval)
    {
        var step = ParseInterval(interval);
        var result = new List<DateTime>();

        for (var current = start; current <= end; current = current.Add(step))
        {
            result.Add(current);
        }

        return result;
    }

    private static List<(DateTime Start, DateTime End)> BuildMissingRanges(
        IReadOnlyList<DateTime> expectedTimestamps,
        SortedDictionary<string, TimeSeriesValue> intervalBucket)
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

    private static TimeSpan ParseInterval(string interval)
    {
        var valuePart = new string(interval.TakeWhile(char.IsDigit).ToArray());
        var unitPart = new string(interval.SkipWhile(char.IsDigit).ToArray()).ToLowerInvariant();

        if (!int.TryParse(valuePart, out var value) || value <= 0)
            throw new InvalidOperationException($"Unsupported interval '{interval}'.");

        return unitPart switch
        {
            "min" => TimeSpan.FromMinutes(value),
            "h" => TimeSpan.FromHours(value),
            "day" => TimeSpan.FromDays(value),
            _ => throw new InvalidOperationException($"Unsupported interval '{interval}'.")
        };
    }

    private static Dictionary<DateTime, TimeSeriesValue> ParseJson(string json)
    {
        using var doc = JsonDocument.Parse(json);

        if (!doc.RootElement.TryGetProperty("values", out var values) || values.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException("Unexpected JSON response: missing 'values' array.");

        var result = new Dictionary<DateTime, TimeSeriesValue>();

        foreach (var item in values.EnumerateArray())
        {
            var dtRaw = item.GetProperty("datetime").GetString()
                ?? throw new InvalidOperationException("Missing datetime.");

            var dt = DateTime.Parse(dtRaw, CultureInfo.InvariantCulture);

            result[dt] = new TimeSeriesValue(
                Datetime: dt,
                Open: ParseDecimal(item, "open"),
                High: ParseDecimal(item, "high"),
                Low: ParseDecimal(item, "low"),
                Close: ParseDecimal(item, "close"));
        }

        return result;
    }

    private static decimal ParseDecimal(JsonElement item, string propertyName)
    {
        var s = item.GetProperty(propertyName).GetString();
        if (string.IsNullOrWhiteSpace(s))
            throw new InvalidOperationException($"Missing '{propertyName}'.");

        return decimal.Parse(s, NumberStyles.Number, CultureInfo.InvariantCulture);
    }

    private static Dictionary<DateTime, TimeSeriesValue> ParseCsv(string csv)
    {
        var result = new Dictionary<DateTime, TimeSeriesValue>();
        var lines = csv.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        if (lines.Length <= 1)
            return result;

        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',', StringSplitOptions.None);
            if (parts.Length < 5) continue;

            var dt = DateTime.Parse(parts[0], CultureInfo.InvariantCulture);

            result[dt] = new TimeSeriesValue(
                Datetime: dt,
                Open: decimal.Parse(parts[1], CultureInfo.InvariantCulture),
                High: decimal.Parse(parts[2], CultureInfo.InvariantCulture),
                Low: decimal.Parse(parts[3], CultureInfo.InvariantCulture),
                Close: decimal.Parse(parts[4], CultureInfo.InvariantCulture));
        }

        return result;
    }

    private static string ToStorageKey(DateTime value) =>
        value.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

    private static DateTime ParseStorageKey(string value) =>
        DateTime.ParseExact(value, "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

    private static class QueryHelpers
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
