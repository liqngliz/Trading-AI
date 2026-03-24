using System.Globalization;
using System.Net.Http;
using System.Text.Json;

namespace Integrations.TwelveData;

public static class TwelveDataSeries
{
    public static Task<Dictionary<DateTime, TimeSeriesValue>> GetSeries(TwelveDataParam param)
        => GetSeries(param, TwelveDataFunctions<TimeSeriesValue>.Default);

    public static async Task<Dictionary<DateTime, TimeSeriesValue>> GetSeries(
        TwelveDataParam param,
        TwelveDataFunctions<TimeSeriesValue> functions)
    {
        ArgumentNullException.ThrowIfNull(param);
        var buildExpectedTimestamps = functions.BuildExpectedTimestamps;
        var buildMissingRanges      = functions.BuildMissingRanges;
        var toStorageKey            = functions.ToStorageKey;
        var parseStorageKey         = functions.ParseStorageKey;

        var cache = await param.Repository
            .GetAsync(param.Symbol, param.Interval, param.CancellationToken)
            .ConfigureAwait(false)
            ?? new TimeSeriesCacheDocument { Symbol = param.Symbol };

        if (!cache.Intervals.TryGetValue(param.Interval, out var intervalBucket))
        {
            intervalBucket = new SortedDictionary<string, TimeSeriesValue>(StringComparer.Ordinal);
            cache.Intervals[param.Interval] = intervalBucket;
        }

        var expectedTimestamps = buildExpectedTimestamps(param.StartDate, param.EndDate, param.Interval);
        Console.WriteLine($"Expected timestamps count: {expectedTimestamps.Count}");
        Console.WriteLine($"Cached timestamps count: {intervalBucket.Count}");
        Console.WriteLine($"Expected last: {expectedTimestamps.LastOrDefault()}");
        var missingRanges = buildMissingRanges(expectedTimestamps, intervalBucket);

        foreach (var missingRange in missingRanges)
        {
            var partialParam = new TwelveDataParam(
                httpClient: param.HttpClient,
                repository: param.Repository,
                apiKey: param.ApiKey,
                symbol: param.Symbol,
                startDate: missingRange.Start,
                endDate: missingRange.End,
                format: param.Format,
                endpoint: param.Endpoint,
                interval: param.Interval,
                outputSize: param.OutputSize,
                cancellationToken: param.CancellationToken);

            var fetched = await GetSeriesFromApiUntilRangeStart(partialParam).ConfigureAwait(false);

            foreach (var kvp in fetched)
                intervalBucket[toStorageKey(kvp.Key)] = kvp.Value;
        }

        var filledMissingCount = FillMissingRanges(expectedTimestamps, intervalBucket);

        if (missingRanges.Count > 0 || filledMissingCount > 0)
        {
            await param.Repository
                .SaveAsync(param.Symbol, param.Interval, cache, param.CancellationToken)
                .ConfigureAwait(false);
        }

        return intervalBucket
            .Where(x =>
            {
                var dt = parseStorageKey(x.Key);
                return dt >= param.StartDate && dt <= param.EndDate;
            })
            .ToDictionary(
                x => parseStorageKey(x.Key),
                x => x.Value);
    }

    private static async Task<Dictionary<DateTime, TimeSeriesValue>> GetSeriesFromApiUntilRangeStart(
        TwelveDataParam param)
    {
        var result = new Dictionary<DateTime, TimeSeriesValue>();
        var requestedStart = param.StartDate;
        var currentEnd = param.EndDate;
        DateTime? lastOldestFetched = null;

        while (currentEnd >= requestedStart)
        {
            var pageParam = new TwelveDataParam(
                httpClient: param.HttpClient,
                repository: param.Repository,
                apiKey: param.ApiKey,
                symbol: param.Symbol,
                startDate: requestedStart,
                endDate: currentEnd,
                format: param.Format,
                endpoint: param.Endpoint,
                interval: param.Interval,
                outputSize: param.OutputSize,
                cancellationToken: param.CancellationToken);

            var page = await GetSeriesFromApi(pageParam).ConfigureAwait(false);

            if (page.Count == 0)
                break;

            foreach (var kvp in page)
                result[kvp.Key] = kvp.Value;

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

    private static async Task<Dictionary<DateTime, TimeSeriesValue>> GetSeriesFromApi(TwelveDataParam param)
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

        var endpointPath = TwelveDataParamExtensions.GetEndpointPath(param.Endpoint);
        var requestUri = new Uri(TwelveDataParamExtensions.QueryHelpers.AddQueryString(endpointPath, query), UriKind.Relative);

        while (true)
        {
            await TwelveDataRateLimiter.WaitForSlotAsync(param.CancellationToken);

            using var response = await param.HttpClient
                .GetAsync(requestUri, param.CancellationToken)
                .ConfigureAwait(false);

            response.EnsureSuccessStatusCode();

            var content = await response.Content
                .ReadAsStringAsync(param.CancellationToken)
                .ConfigureAwait(false);

            try
            {
                return param.Format == TwelveDataFormat.Json
                    ? ParseJson(content)
                    : ParseCsv(content);
            }
            catch (TwelveDataRateLimitException ex)
            {
                Console.WriteLine($"  [429] {ex.Message} Waiting {TwelveDataRateLimiter.RetryDelay.TotalSeconds:F0}s before retry...");
                await Task.Delay(TwelveDataRateLimiter.RetryDelay, param.CancellationToken);
            }
        }
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
                var hasValue = intervalBucket.TryGetValue(TwelveDataParamExtensions.ToStorageKey(timestamp), out var value);
                return new { Timestamp = timestamp, Index = index, HasValue = hasValue, Value = value };
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
            intervalBucket[TwelveDataParamExtensions.ToStorageKey(timestamp)] = CreateFilledValue(timestamp, firstExisting.Value);
            filledCount++;
        }

        var lastKnown = firstExisting.Value;

        for (var i = firstExisting.Index + 1; i < expectedTimestamps.Count; i++)
        {
            var timestamp = expectedTimestamps[i];
            var storageKey = TwelveDataParamExtensions.ToStorageKey(timestamp);

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

    internal static Dictionary<DateTime, TimeSeriesValue> ParseJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.TryGetProperty("status", out var statusEl) &&
            string.Equals(statusEl.GetString(), "error", StringComparison.OrdinalIgnoreCase))
        {
            var code = root.TryGetProperty("code", out var codeEl) && codeEl.ValueKind == JsonValueKind.Number
                ? codeEl.GetInt32() : 0;
            var message = root.TryGetProperty("message", out var msgEl) ? msgEl.GetString() : null;

            if (code == 400)
                return new Dictionary<DateTime, TimeSeriesValue>();

            if (code == 429)
                throw new TwelveDataRateLimitException(message ?? "Rate limit exceeded.");

            if (code == 404)
                throw new TwelveDataSymbolUnavailableException(message ?? "Symbol not available on current plan.");

            throw new InvalidOperationException($"TwelveData API error {code}: {message}");
        }

        if (!root.TryGetProperty("values", out var values) || values.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException("Unexpected JSON response: missing 'values' array.");

        var result = new Dictionary<DateTime, TimeSeriesValue>();

        foreach (var item in values.EnumerateArray())
        {
            var dtRaw = item.GetProperty("datetime").GetString()
                ?? throw new InvalidOperationException("Missing datetime.");

            var dt = DateTime.Parse(dtRaw, CultureInfo.InvariantCulture);

            result[dt] = new TimeSeriesValue(
                Datetime: dt,
                Open: TwelveDataParamExtensions.ParseDecimal(item, "open"),
                High: TwelveDataParamExtensions.ParseDecimal(item, "high"),
                Low: TwelveDataParamExtensions.ParseDecimal(item, "low"),
                Close: TwelveDataParamExtensions.ParseDecimal(item, "close"),
                IsFilled: false);
        }

        return result;
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
}
