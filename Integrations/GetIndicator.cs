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

        var cacheKey = param.Symbol;
        var bucketKey = $"{getEndpointPath(param.Endpoint)}/{param.Interval}";

        var cache = await param.IndicatorRepository
            .GetAsync(cacheKey, param.CancellationToken)
            .ConfigureAwait(false)
            ?? new IndicatorCacheDocument { Symbol = param.Symbol };

        if (!cache.Data.TryGetValue(bucketKey, out var bucket))
        {
            bucket = new SortedDictionary<string, IndicatorValue>(StringComparer.Ordinal);
            cache.Data[bucketKey] = bucket;
        }

        var expectedTimestamps = buildExpectedTimestamps(param.StartDate, param.EndDate, param.Interval);
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

        var filledMissingCount = FillMissingIndicatorRanges(expectedTimestamps, bucket);

        if (missingRanges.Count > 0 || filledMissingCount > 0)
        {
            await param.IndicatorRepository
                .SaveAsync(cacheKey, cache, param.CancellationToken)
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

        using var response = await param.HttpClient
            .GetAsync(requestUri, param.CancellationToken)
            .ConfigureAwait(false);

        response.EnsureSuccessStatusCode();

        var content = await response.Content
            .ReadAsStringAsync(param.CancellationToken)
            .ConfigureAwait(false);

        return ParseIndicatorJson(content, GetIndicatorFieldName(param.Endpoint));
    }

    internal static int FillMissingIndicatorRanges(
        IReadOnlyList<DateTime> expectedTimestamps,
        SortedDictionary<string, IndicatorValue> bucket)
    {
        if (expectedTimestamps.Count == 0)
            return 0;

        // Find seed: earliest real (non-filled) entry in the bucket, regardless of expectedTimestamp alignment.
        // Bucket is sorted by key (= sorted chronologically via "yyyy-MM-dd HH:mm:ss" format).
        var seedEntry = bucket.FirstOrDefault(x => !x.Value.IsFilled);
        if (seedEntry.Key is null)
            return 0;

        var lastKnownValue = seedEntry.Value.Value;
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

        if (TwelveDataParamExtensions.IsNoDataResponse(doc.RootElement))
            return new Dictionary<DateTime, decimal>();

        // Any other API error (e.g. indicator not calculable for this range) — treat as no data.
        if (doc.RootElement.TryGetProperty("status", out var statusEl) &&
            string.Equals(statusEl.GetString(), "error", StringComparison.OrdinalIgnoreCase))
            return new Dictionary<DateTime, decimal>();

        if (!doc.RootElement.TryGetProperty("values", out var values) || values.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException("Unexpected JSON response: missing 'values' array.");

        var result = new Dictionary<DateTime, decimal>();

        foreach (var item in values.EnumerateArray())
        {
            var dtRaw = item.GetProperty("datetime").GetString()
                ?? throw new InvalidOperationException("Missing datetime.");

            var valueStr = item.GetProperty(fieldName).GetString();
            if (string.IsNullOrWhiteSpace(valueStr) ||
                valueStr.Equals("NaN", StringComparison.OrdinalIgnoreCase))
                continue;

            var dt = DateTime.Parse(dtRaw, CultureInfo.InvariantCulture);
            result[dt] = decimal.Parse(valueStr, NumberStyles.Number, CultureInfo.InvariantCulture);
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
