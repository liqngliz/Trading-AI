using System.Globalization;
using System.Net.Http;
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

public sealed record TwelveTimeSeriesParam
{
    public HttpClient HttpClient { get; }
    public string ApiKey { get; }
    public string Symbol { get; }
    public DateTime StartDate { get; }
    public DateTime EndDate { get; }
    public TwelveDataFormat Format { get; }
    public string Interval { get; }
    public CancellationToken CancellationToken { get; }

    public TwelveTimeSeriesParam(
        HttpClient httpClient,
        string apiKey,
        string symbol,
        DateTime startDate,
        DateTime endDate,
        TwelveDataFormat format,
        string interval = "4h",
        CancellationToken cancellationToken = default)
    {
        if (httpClient is null) throw new ArgumentNullException(nameof(httpClient));
        if (httpClient.BaseAddress is null)
            throw new InvalidOperationException("HttpClient.BaseAddress must be set (use IHttpClientFactory configuration).");
        if (string.IsNullOrWhiteSpace(apiKey)) throw new ArgumentException("API key is required.", nameof(apiKey));
        if (string.IsNullOrWhiteSpace(symbol)) throw new ArgumentException("Symbol is required.", nameof(symbol));
        if (startDate > endDate) throw new ArgumentException("startDate must be <= endDate.", nameof(startDate));
        if (string.IsNullOrWhiteSpace(interval)) throw new ArgumentException("Interval is required.", nameof(interval));

        HttpClient = httpClient;
        ApiKey = apiKey;
        Symbol = symbol;
        StartDate = startDate;
        EndDate = endDate;
        Format = format;
        Interval = interval;
        CancellationToken = cancellationToken;
    }
}

public static class TwelveTimeSeriesParamExtensions
{
    public static async Task<Dictionary<DateTime, TimeSeriesValue>> GetSeries(
        this TwelveTimeSeriesParam param)
    {
        ArgumentNullException.ThrowIfNull(param);

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
            new("format", formatString)
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

        if (lines.Length <= 1) return result;

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

    private static class QueryHelpers
    {
        public static string AddQueryString(string uri, IEnumerable<KeyValuePair<string?, string?>> queryString)
        {
            var hasQuery = uri.Contains('?', StringComparison.Ordinal);
            var sb = new System.Text.StringBuilder(uri);

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
