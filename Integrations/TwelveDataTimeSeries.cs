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
    decimal Close,
    bool IsFilled);

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
        Console.WriteLine($"Expected timestamps count: {expectedTimestamps.Count}");
        Console.WriteLine($"Cached timestamps count: {intervalBucket.Count}");
        Console.WriteLine($"Expected last: {expectedTimestamps.LastOrDefault()}");
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

            var fetched = await GetSeriesFromApiUntilRangeStart(partialParam).ConfigureAwait(false);

            foreach (var kvp in fetched)
            {
                intervalBucket[ToStorageKey(kvp.Key)] = kvp.Value;
            }
        }

        var filledMissingCount = FillMissingRanges(
            expectedTimestamps,
            intervalBucket);

        if (missingRanges.Count > 0 || filledMissingCount > 0)
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

        var step = ParseInterval(param.Interval);
        var now = DateTime.UtcNow;

        var nonFilledKeys = intervalBucket
            .Where(x => !x.Value.IsFilled)
            .Select(x => ParseStorageKey(x.Key))
            .OrderBy(x => x)
            .ToList();

        if (nonFilledKeys.Count == 0)
        {
            var uncachedParam = new TwelveTimeSeriesParam(
                httpClient: param.HttpClient,
                repository: param.Repository,
                apiKey: param.ApiKey,
                symbol: param.Symbol,
                startDate: param.StartDate,
                endDate: now,
                format: param.Format,
                interval: param.Interval,
                outputSize: param.OutputSize,
                cancellationToken: param.CancellationToken);

            return await uncachedParam.GetSeries().ConfigureAwait(false);
        }

        var earliestNonFilled = nonFilledKeys[0];
        var latestNonFilled = nonFilledKeys[^1];

        var backwardEnd = earliestNonFilled.AddTicks(-1);
        if (param.StartDate <= backwardEnd)
        {
            var backwardParam = new TwelveTimeSeriesParam(
                httpClient: param.HttpClient,
                repository: param.Repository,
                apiKey: param.ApiKey,
                symbol: param.Symbol,
                startDate: param.StartDate,
                endDate: backwardEnd,
                format: param.Format,
                interval: param.Interval,
                outputSize: param.OutputSize,
                cancellationToken: param.CancellationToken);

            await backwardParam.GetSeries().ConfigureAwait(false);
        }

        var forwardStart = latestNonFilled.Add(step);
        if (forwardStart <= now)
        {
            var forwardParam = new TwelveTimeSeriesParam(
                httpClient: param.HttpClient,
                repository: param.Repository,
                apiKey: param.ApiKey,
                symbol: param.Symbol,
                startDate: forwardStart,
                endDate: now,
                format: param.Format,
                interval: param.Interval,
                outputSize: param.OutputSize,
                cancellationToken: param.CancellationToken);

            await forwardParam.GetSeries().ConfigureAwait(false);
        }

        cache = await param.Repository
            .GetAsync(cacheKey, param.CancellationToken)
            .ConfigureAwait(false)
            ?? cache;

        if (!cache.Intervals.TryGetValue(param.Interval, out intervalBucket))
        {
            return new Dictionary<DateTime, TimeSeriesValue>();
        }

        return intervalBucket.ToDictionary(
            x => ParseStorageKey(x.Key),
            x => x.Value);
    }

    private static async Task<Dictionary<DateTime, TimeSeriesValue>> GetSeriesFromApiUntilRangeStart(
        TwelveTimeSeriesParam param)
    {
        var result = new Dictionary<DateTime, TimeSeriesValue>();
        var requestedStart = param.StartDate;
        var currentEnd = param.EndDate;
        DateTime? lastOldestFetched = null;

        while (currentEnd >= requestedStart)
        {
            var pageParam = new TwelveTimeSeriesParam(
                httpClient: param.HttpClient,
                repository: param.Repository,
                apiKey: param.ApiKey,
                symbol: param.Symbol,
                startDate: requestedStart,
                endDate: currentEnd,
                format: param.Format,
                interval: param.Interval,
                outputSize: param.OutputSize,
                cancellationToken: param.CancellationToken);

            var page = await GetSeriesFromApi(pageParam).ConfigureAwait(false);

            if (page.Count == 0)
                break;

            foreach (var kvp in page)
            {
                result[kvp.Key] = kvp.Value;
            }

            var oldestFetched = page.Keys.Min();

            if (oldestFetched <= requestedStart)
                break;

            if (lastOldestFetched.HasValue && oldestFetched >= lastOldestFetched.Value)
                break;

            lastOldestFetched = oldestFetched;
            currentEnd = oldestFetched.AddTicks(-1);
        }

        return result;
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

    internal static List<(DateTime Start, DateTime End)> BuildMissingRanges(
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

    internal static int FillMissingRanges(
        IReadOnlyList<DateTime> expectedTimestamps,
        SortedDictionary<string, TimeSeriesValue> intervalBucket)
    {
        if (expectedTimestamps.Count == 0)
            return 0;

        var existingValues = expectedTimestamps
            .Select((timestamp, index) =>
            {
                var hasValue = intervalBucket.TryGetValue(ToStorageKey(timestamp), out var value);
                return new
                {
                    Timestamp = timestamp,
                    Index = index,
                    HasValue = hasValue,
                    Value = value
                };
            })
            .Where(x => x.HasValue)
            .ToList();

        if (existingValues.Count == 0)
            return 0;

        var firstExisting = existingValues[0];
        var filledCount = 0;

        for (var i = 0; i < firstExisting.Index; i++)
        {
            var timestamp = expectedTimestamps[i];
            intervalBucket[ToStorageKey(timestamp)] = CreateFilledValue(timestamp, firstExisting.Value);
            filledCount++;
        }

        var lastKnown = firstExisting.Value;

        for (var i = firstExisting.Index + 1; i < expectedTimestamps.Count; i++)
        {
            var timestamp = expectedTimestamps[i];
            var storageKey = ToStorageKey(timestamp);

            if (intervalBucket.TryGetValue(storageKey, out var existingValue))
            {
                lastKnown = existingValue;
                continue;
            }

            intervalBucket[storageKey] = CreateFilledValue(timestamp, lastKnown);
            filledCount++;
        }

        return filledCount;
    }

    internal static TimeSeriesValue CreateFilledValue(DateTime timestamp, TimeSeriesValue source) =>
        new(
            Datetime: timestamp,
            Open: source.Open,
            High: source.High,
            Low: source.Low,
            Close: source.Close,
            IsFilled: true);

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

    internal static Dictionary<DateTime, TimeSeriesValue> ParseJson(string json)
    {
        using var doc = JsonDocument.Parse(json);

        if (IsNoDataResponse(doc.RootElement))
            return new Dictionary<DateTime, TimeSeriesValue>();

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
                Close: ParseDecimal(item, "close"),
                IsFilled: false);
        }

        return result;
    }

    internal static decimal ParseDecimal(JsonElement item, string propertyName)
    {
        var s = item.GetProperty(propertyName).GetString();
        if (string.IsNullOrWhiteSpace(s))
            throw new InvalidOperationException($"Missing '{propertyName}'.");

        return decimal.Parse(s, NumberStyles.Number, CultureInfo.InvariantCulture);
    }

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


    internal static Dictionary<DateTime, TimeSeriesValue> ParseCsv(string csv)
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
                Close: decimal.Parse(parts[4], CultureInfo.InvariantCulture),
                IsFilled: false);
        }

        return result;
    }

    internal static string ToStorageKey(DateTime value) =>
        value.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

    internal static DateTime ParseStorageKey(string value) =>
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
