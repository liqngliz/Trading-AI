using System;
using System.Collections.Generic;
using System.Linq;
namespace Integrations.TwelveData;


public static class TimeSeriesValueExtensions
{
    public static Dictionary<DateTime, TimeSeriesValue> TrimLeadingFilledCandles(
        this IReadOnlyDictionary<DateTime, TimeSeriesValue> candles)
    {
        ArgumentNullException.ThrowIfNull(candles);

        var ordered = candles
            .OrderBy(x => x.Key)
            .ToList();

        var firstNonFilledIndex = ordered.FindIndex(x => !x.Value.IsFilled);

        if (firstNonFilledIndex < 0)
        {
            return new Dictionary<DateTime, TimeSeriesValue>();
        }

        return ordered
            .Skip(firstNonFilledIndex)
            .ToDictionary(x => x.Key, x => x.Value);
    }

    public static Dictionary<DateTime, TimeSeriesValue> TrimTrailingFilledCandles(
        this IReadOnlyDictionary<DateTime, TimeSeriesValue> candles)
    {
        ArgumentNullException.ThrowIfNull(candles);

        var ordered = candles
            .OrderBy(x => x.Key)
            .ToList();

        var lastNonFilledIndex = ordered.FindLastIndex(x => !x.Value.IsFilled);

        if (lastNonFilledIndex < 0)
        {
            return new Dictionary<DateTime, TimeSeriesValue>();
        }

        return ordered
            .Take(lastNonFilledIndex + 1)
            .ToDictionary(x => x.Key, x => x.Value);
    }

    public static Dictionary<DateTime, TimeSeriesValue> TrimLeadingAndTrailingFilledCandles(
        this IReadOnlyDictionary<DateTime, TimeSeriesValue> candles)
    {
        ArgumentNullException.ThrowIfNull(candles);

        var ordered = candles
            .OrderBy(x => x.Key)
            .ToList();

        if (ordered.Count == 0)
        {
            return new Dictionary<DateTime, TimeSeriesValue>();
        }

        var firstNonFilledIndex = ordered.FindIndex(x => !x.Value.IsFilled);
        var lastNonFilledIndex = ordered.FindLastIndex(x => !x.Value.IsFilled);

        if (firstNonFilledIndex < 0 || lastNonFilledIndex < 0)
        {
            return new Dictionary<DateTime, TimeSeriesValue>();
        }

        return ordered
            .Skip(firstNonFilledIndex)
            .Take(lastNonFilledIndex - firstNonFilledIndex + 1)
            .ToDictionary(x => x.Key, x => x.Value);
    }
}