using System.Diagnostics;

namespace Integrations.TwelveData;

public static class ValueExtensions
{
    // ── Generic core ──────────────────────────────────────────────────────────

    public static Dictionary<DateTime, TValue> TrimLeading<TValue>(
        this IReadOnlyDictionary<DateTime, TValue> values,
        Func<TValue, bool> shouldTrim)
    {
        ArgumentNullException.ThrowIfNull(values);

        var ordered = values.OrderBy(x => x.Key).ToList();
        var firstKeptIndex = ordered.FindIndex(x => !shouldTrim(x.Value));

        if (firstKeptIndex < 0)
            return new Dictionary<DateTime, TValue>();

        return ordered
            .Skip(firstKeptIndex)
            .ToDictionary(x => x.Key, x => x.Value);
    }

    public static Dictionary<DateTime, TValue> TrimTrailing<TValue>(
        this IReadOnlyDictionary<DateTime, TValue> values,
        Func<TValue, bool> shouldTrim)
    {
        ArgumentNullException.ThrowIfNull(values);

        var ordered = values.OrderBy(x => x.Key).ToList();
        var lastKeptIndex = ordered.FindLastIndex(x => !shouldTrim(x.Value));

        if (lastKeptIndex < 0)
            return new Dictionary<DateTime, TValue>();

        return ordered
            .Take(lastKeptIndex + 1)
            .ToDictionary(x => x.Key, x => x.Value);
    }

    public static Dictionary<DateTime, TValue> TrimLeadingAndTrailing<TValue>(
        this IReadOnlyDictionary<DateTime, TValue> values,
        Func<TValue, bool> shouldTrim)
    {
        ArgumentNullException.ThrowIfNull(values);

        var sw = Stopwatch.StartNew();

        var ordered = values.OrderBy(x => x.Key).ToList();

        if (ordered.Count == 0)
            return new Dictionary<DateTime, TValue>();

        int firstKeptIndex = 0;
        while (firstKeptIndex < ordered.Count && shouldTrim(ordered[firstKeptIndex].Value))
            firstKeptIndex++;

        int lastKeptIndex = ordered.Count - 1;
        while (lastKeptIndex > firstKeptIndex && shouldTrim(ordered[lastKeptIndex].Value))
            lastKeptIndex--;

        if (firstKeptIndex >= ordered.Count || shouldTrim(ordered[firstKeptIndex].Value))
            return new Dictionary<DateTime, TValue>();

        var result = ordered
            .Skip(firstKeptIndex)
            .Take(lastKeptIndex - firstKeptIndex + 1)
            .ToDictionary(x => x.Key, x => x.Value);

        Console.WriteLine($"[TrimLeadingAndTrailing] {values.Count} → {result.Count} entries in {sw.ElapsedMilliseconds}ms");
        return result;
    }

    // ── TimeSeriesValue convenience ───────────────────────────────────────────

    public static Dictionary<DateTime, TimeSeriesValue> TrimLeadingFilledCandles(
        this IReadOnlyDictionary<DateTime, TimeSeriesValue> candles) =>
        candles.TrimLeading(v => v.IsFilled);

    public static Dictionary<DateTime, TimeSeriesValue> TrimTrailingFilledCandles(
        this IReadOnlyDictionary<DateTime, TimeSeriesValue> candles) =>
        candles.TrimTrailing(v => v.IsFilled);

    public static Dictionary<DateTime, TimeSeriesValue> TrimLeadingAndTrailingFilledCandles(
        this IReadOnlyDictionary<DateTime, TimeSeriesValue> candles) =>
        candles.TrimLeadingAndTrailing(v => v.IsFilled);

    // ── IndicatorValue convenience ────────────────────────────────────────────

    public static Dictionary<DateTime, IndicatorValue> TrimLeadingFilledIndicators(
        this IReadOnlyDictionary<DateTime, IndicatorValue> values) =>
        values.TrimLeading(v => v.IsFilled);

    public static Dictionary<DateTime, IndicatorValue> TrimTrailingFilledIndicators(
        this IReadOnlyDictionary<DateTime, IndicatorValue> values) =>
        values.TrimTrailing(v => v.IsFilled);

    public static Dictionary<DateTime, IndicatorValue> TrimLeadingAndTrailingFilledIndicators(
        this IReadOnlyDictionary<DateTime, IndicatorValue> values) =>
        values.TrimLeadingAndTrailing(v => v.IsFilled);

    // ── decimal convenience ───────────────────────────────────────────────────

    public static Dictionary<DateTime, decimal> TrimLeadingZeroValues(
        this IReadOnlyDictionary<DateTime, decimal> values) =>
        values.TrimLeading(v => v == 0m);

    public static Dictionary<DateTime, decimal> TrimTrailingZeroValues(
        this IReadOnlyDictionary<DateTime, decimal> values) =>
        values.TrimTrailing(v => v == 0m);

    public static Dictionary<DateTime, decimal> TrimLeadingAndTrailingZeroValues(
        this IReadOnlyDictionary<DateTime, decimal> values) =>
        values.TrimLeadingAndTrailing(v => v == 0m);
}
