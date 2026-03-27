using System.Globalization;
using System.Net.Http;
using System.Text.Json;

namespace Integrations.TwelveData;

public static class TwelveDataIndicator
{
    /// <summary>
    /// Fetches a volume indicator (AD, ADOSC, OBV, RVOL) for the given date range.
    /// Returns a dictionary of timestamp → indicator value.
    /// TwelveData default period values are used for ADOSC and RVOL.
    /// When <see cref="TwelveDataParam.IndicatorRepository"/> is set, results are cached and
    /// only missing date ranges are fetched from the API.
    /// </summary>
    public static Task<Dictionary<DateTime, IndicatorValue>> GetIndicator(TwelveDataParam param)
        => GetIndicator(param, TwelveDataFunctions<IndicatorValue>.Default);

    public static async Task<Dictionary<DateTime, IndicatorValue>> GetIndicator(
        TwelveDataParam param,
        TwelveDataFunctions<IndicatorValue> functions)
    {
        ArgumentNullException.ThrowIfNull(param);
        var getEndpointPath         = functions.GetEndpointPath;
        var buildExpectedTimestamps = functions.BuildExpectedTimestamps;
        var buildMissingRanges      = functions.BuildMissingRanges;
        var toStorageKey            = functions.ToStorageKey;
        var parseStorageKey         = functions.ParseStorageKey;

        if (param.Endpoint == TwelveDataEndpoint.TimeSeries)
            throw new InvalidOperationException(
                "GetIndicator() is for volume indicator endpoints. Use GetSeries() for TimeSeries.");

        if (param.IndicatorRepository is null)
        {
            var raw = await GetIndicatorFromApiUntilRangeStart(param).ConfigureAwait(false);
            return raw.ToDictionary(
                kvp => kvp.Key,
                kvp => new IndicatorValue(kvp.Key, kvp.Value, IsFilled: false));
        }

        var bucketKey = $"{getEndpointPath(param.Endpoint)}/{param.Interval}";

        var cache = await param.IndicatorRepository
            .GetAsync(param.Symbol, bucketKey, param.CancellationToken)
            .ConfigureAwait(false)
            ?? new IndicatorCacheDocument { Symbol = param.Symbol };

        if (cache.UnavailableBuckets.Contains(bucketKey))
        {
            Console.WriteLine($"  [GetIndicator] {param.Symbol}/{bucketKey}: previously marked unavailable, skipping.");
            return new Dictionary<DateTime, IndicatorValue>();
        }

        if (!cache.Data.TryGetValue(bucketKey, out var bucket))
        {
            bucket = new SortedDictionary<string, IndicatorValue>(StringComparer.Ordinal);
            cache.Data[bucketKey] = bucket;
        }

        // Compute effective fetch start from the latest real cached entry.
        var step = TwelveDataParamExtensions.ParseInterval(param.Interval);
        var effectiveStart = param.StartDate;
        foreach (var kv in bucket)
            if (!kv.Value.IsFilled) effectiveStart = kv.Value.Datetime.Add(step);

        Console.WriteLine($"  [GetIndicator] {param.Symbol}/{bucketKey}: {bucket.Count} cached. Fetching from {effectiveStart:yyyy-MM-dd HH:mm:ss}");

        if (effectiveStart > param.EndDate)
        {
            Console.WriteLine($"  [GetIndicator] {param.Symbol}/{bucketKey}: cache is up to date.");
            return bucket
                .Where(x => { var dt = parseStorageKey(x.Key); return dt >= param.StartDate && dt <= param.EndDate; })
                .ToDictionary(x => parseStorageKey(x.Key), x => x.Value);
        }

        var expectedTimestamps = buildExpectedTimestamps(effectiveStart, param.EndDate, param.Interval);
        var missingRanges = buildMissingRanges(expectedTimestamps, bucket);

        foreach (var missingRange in missingRanges)
        {
            var rangeParam = new TwelveDataParam(
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
                indicatorRepository: param.IndicatorRepository,
                cancellationToken: param.CancellationToken);

            var fetched = await GetIndicatorFromApiUntilRangeStart(rangeParam).ConfigureAwait(false);
            foreach (var kvp in fetched)
                bucket[toStorageKey(kvp.Key)] = new IndicatorValue(kvp.Key, kvp.Value, IsFilled: false);
        }

        // If we fetched but got no real data at all, mark the bucket as permanently unavailable.
        if (missingRanges.Count > 0 && !bucket.Values.Any(v => !v.IsFilled))
        {
            cache.UnavailableBuckets.Add(bucketKey);
            Console.WriteLine($"  [GetIndicator] {param.Symbol}/{bucketKey}: API returned no usable data, marked as unavailable.");
            await param.IndicatorRepository
                .SaveAsync(param.Symbol, bucketKey, cache, param.CancellationToken)
                .ConfigureAwait(false);
            return new Dictionary<DateTime, IndicatorValue>();
        }

        var filledMissingCount = FillMissingIndicatorRanges(expectedTimestamps, bucket);

        if (missingRanges.Count > 0 || filledMissingCount > 0)
        {
            await param.IndicatorRepository
                .SaveAsync(param.Symbol, bucketKey, cache, param.CancellationToken)
                .ConfigureAwait(false);
        }

        return bucket
            .Where(x =>
            {
                var dt = parseStorageKey(x.Key);
                return dt >= param.StartDate && dt <= param.EndDate;
            })
            .ToDictionary(
                x => parseStorageKey(x.Key),
                x => x.Value);
    }

    private static async Task<Dictionary<DateTime, decimal>> GetIndicatorFromApiUntilRangeStart(
        TwelveDataParam param)
    {
        var result = new Dictionary<DateTime, decimal>();
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

            var page = await GetIndicatorFromApi(pageParam).ConfigureAwait(false);

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

    private static async Task<Dictionary<DateTime, decimal>> GetIndicatorFromApi(TwelveDataParam param)
    {
        var query = new List<KeyValuePair<string?, string?>>
        {
            new("apikey", param.ApiKey),
            new("interval", param.Interval),
            new("symbol", param.Symbol),
            new("start_date", param.StartDate.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture)),
            new("end_date", param.EndDate.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture)),
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
                return ParseIndicatorJson(content, GetIndicatorFieldName(param.Endpoint));
            }
            catch (TwelveDataRateLimitException ex)
            {
                Console.WriteLine($"  [429] {ex.Message} Waiting {TwelveDataRateLimiter.RetryDelay.TotalSeconds:F0}s before retry...");
                await Task.Delay(TwelveDataRateLimiter.RetryDelay, param.CancellationToken);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [GetIndicatorFromApi] Error parsing response for symbol={param.Symbol}, endpoint={param.Endpoint}: {ex.Message}");
                throw;
            }
        }
    }

    internal static int FillMissingIndicatorRanges(
        IReadOnlyList<DateTime> expectedTimestamps,
        SortedDictionary<string, IndicatorValue> bucket)
    {
        if (expectedTimestamps.Count == 0)
            return 0;

        // Seed from the latest real (non-filled) entry at or before the first expected
        // timestamp so carry-forward uses the most recent known value, not the oldest.
        var firstExpectedKey = TwelveDataParamExtensions.ToStorageKey(expectedTimestamps[0]);
        decimal? seed = null;
        foreach (var kv in bucket)
        {
            if (kv.Value.IsFilled) continue;
            if (string.Compare(kv.Key, firstExpectedKey, StringComparison.Ordinal) > 0) break;
            seed = kv.Value.Value;
        }

        if (seed is null)
            return 0;

        var lastKnownValue = seed.Value;
        var filledCount = 0;

        foreach (var timestamp in expectedTimestamps)
        {
            var storageKey = TwelveDataParamExtensions.ToStorageKey(timestamp);
            if (bucket.TryGetValue(storageKey, out var existing) && !existing.IsFilled)
            {
                lastKnownValue = existing.Value;
                continue;
            }
            bucket[storageKey] = new IndicatorValue(timestamp, lastKnownValue, IsFilled: true);
            filledCount++;
        }

        return filledCount;
    }

    internal static Dictionary<DateTime, decimal> ParseIndicatorJson(string json, string fieldName)
    {
        using var doc = JsonDocument.Parse(json);

        if (doc.RootElement.TryGetProperty("status", out var statusEl) &&
            string.Equals(statusEl.GetString(), "error", StringComparison.OrdinalIgnoreCase))
        {
            var code = doc.RootElement.TryGetProperty("code", out var codeEl) && codeEl.ValueKind == JsonValueKind.Number
                ? codeEl.GetInt32() : 0;
            var message = doc.RootElement.TryGetProperty("message", out var msgEl) ? msgEl.GetString() : null;

            if (code == 429)
                throw new TwelveDataRateLimitException(message ?? "Rate limit exceeded.");

            // 400 (no data, indicator not calculable) and any other error — treat as no data.
            return new Dictionary<DateTime, decimal>();
        }

        if (!doc.RootElement.TryGetProperty("values", out var values) || values.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException("Unexpected JSON response: missing 'values' array.");

        var items = values.EnumerateArray().ToList();
        var result = new Dictionary<DateTime, decimal>();

        foreach (var item in items)
        {
            var dtRaw = item.GetProperty("datetime").GetString()
                ?? throw new InvalidOperationException("Missing datetime.");

            var valueStr = item.GetProperty(fieldName).GetString();
            if (string.IsNullOrWhiteSpace(valueStr) ||
                valueStr.Equals("NaN", StringComparison.OrdinalIgnoreCase) ||
                valueStr.Equals("+Inf", StringComparison.OrdinalIgnoreCase) ||
                valueStr.Equals("-Inf", StringComparison.OrdinalIgnoreCase) ||
                valueStr.Equals("Inf", StringComparison.OrdinalIgnoreCase))
                continue;
            
            try
            {
                var dt = DateTime.Parse(dtRaw, CultureInfo.InvariantCulture);
                result[dt] = decimal.Parse(valueStr, NumberStyles.Number, CultureInfo.InvariantCulture);
            }
            catch (FormatException ex)
            {
                Console.WriteLine($"  [ParseIndicatorJson] Failed to parse entry: datetime={dtRaw}, value={valueStr} — {ex.Message}");
                Console.WriteLine($"  [ParseIndicatorJson] Full values array ({items.Count} items):");
                foreach (var i in items)
                    Console.WriteLine($"    {i}");
                throw;
            }
        }

        return result;
    }

    private static string GetIndicatorFieldName(TwelveDataEndpoint endpoint) => endpoint switch
    {
        TwelveDataEndpoint.Ad => "ad",
        TwelveDataEndpoint.Adosc => "adosc",
        TwelveDataEndpoint.Obv => "obv",
        TwelveDataEndpoint.Rvol => "rvol",
        _ => throw new ArgumentOutOfRangeException(nameof(endpoint), endpoint, null)
    };
}
