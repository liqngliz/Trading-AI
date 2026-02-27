using Integrations.TwelveData;
using Integrations.Tests.Helpers;

namespace Integrations.Tests;

public class TimeSeriesValueExtensionsTests
{
    private static readonly DateTime T0 = new DateTime(2024, 1, 1, 0, 0, 0);
    private static readonly DateTime T1 = new DateTime(2024, 1, 1, 4, 0, 0);
    private static readonly DateTime T2 = new DateTime(2024, 1, 1, 8, 0, 0);
    private static readonly DateTime T3 = new DateTime(2024, 1, 1, 12, 0, 0);
    private static readonly DateTime T4 = new DateTime(2024, 1, 1, 16, 0, 0);

    // ── TrimLeadingFilledCandles ──────────────────────────────────────────────

    [Fact]
    public void TrimLeading_NullInput_ThrowsArgumentNullException()
    {
        IReadOnlyDictionary<DateTime, TimeSeriesValue>? candles = null;
        Assert.Throws<ArgumentNullException>(() => candles!.TrimLeadingFilledCandles());
    }

    [Fact]
    public void TrimLeading_AllReal_ReturnsSameCount()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.RealCandle(T0),
            [T1] = TimeSeriesFixtures.RealCandle(T1),
        };
        Assert.Equal(2, candles.TrimLeadingFilledCandles().Count);
    }

    [Fact]
    public void TrimLeading_AllFilled_ReturnsEmpty()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.FilledCandle(T0),
            [T1] = TimeSeriesFixtures.FilledCandle(T1),
        };
        Assert.Empty(candles.TrimLeadingFilledCandles());
    }

    [Fact]
    public void TrimLeading_EmptyDictionary_ReturnsEmpty()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>();
        Assert.Empty(candles.TrimLeadingFilledCandles());
    }

    [Fact]
    public void TrimLeading_LeadingFilledThenReal_RemovesLeadingFilled()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.FilledCandle(T0), // leading — removed
            [T1] = TimeSeriesFixtures.FilledCandle(T1), // leading — removed
            [T2] = TimeSeriesFixtures.RealCandle(T2),
            [T3] = TimeSeriesFixtures.RealCandle(T3),
        };
        var result = candles.TrimLeadingFilledCandles();
        Assert.Equal(2, result.Count);
        Assert.DoesNotContain(T0, result.Keys);
        Assert.DoesNotContain(T1, result.Keys);
    }

    [Fact]
    public void TrimLeading_FilledInMiddle_NotRemoved()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.RealCandle(T0),
            [T1] = TimeSeriesFixtures.FilledCandle(T1), // middle filled — kept
            [T2] = TimeSeriesFixtures.RealCandle(T2),
        };
        var result = candles.TrimLeadingFilledCandles();
        Assert.Equal(3, result.Count);
        Assert.Contains(T1, result.Keys);
    }

    [Fact]
    public void TrimLeading_SingleFilledCandle_ReturnsEmpty()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.FilledCandle(T0)
        };
        Assert.Empty(candles.TrimLeadingFilledCandles());
    }

    [Fact]
    public void TrimLeading_SingleRealCandle_ReturnsThatCandle()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.RealCandle(T0)
        };
        var result = candles.TrimLeadingFilledCandles();
        Assert.Single(result);
        Assert.Contains(T0, result.Keys);
    }

    // ── TrimTrailingFilledCandles ─────────────────────────────────────────────

    [Fact]
    public void TrimTrailing_NullInput_ThrowsArgumentNullException()
    {
        IReadOnlyDictionary<DateTime, TimeSeriesValue>? candles = null;
        Assert.Throws<ArgumentNullException>(() => candles!.TrimTrailingFilledCandles());
    }

    [Fact]
    public void TrimTrailing_AllReal_ReturnsSameCount()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.RealCandle(T0),
            [T1] = TimeSeriesFixtures.RealCandle(T1),
        };
        Assert.Equal(2, candles.TrimTrailingFilledCandles().Count);
    }

    [Fact]
    public void TrimTrailing_AllFilled_ReturnsEmpty()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.FilledCandle(T0),
            [T1] = TimeSeriesFixtures.FilledCandle(T1),
        };
        Assert.Empty(candles.TrimTrailingFilledCandles());
    }

    [Fact]
    public void TrimTrailing_EmptyDictionary_ReturnsEmpty()
    {
        Assert.Empty(new Dictionary<DateTime, TimeSeriesValue>().TrimTrailingFilledCandles());
    }

    [Fact]
    public void TrimTrailing_TrailingFilledRemoved()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.RealCandle(T0),
            [T1] = TimeSeriesFixtures.RealCandle(T1),
            [T2] = TimeSeriesFixtures.FilledCandle(T2), // trailing — removed
            [T3] = TimeSeriesFixtures.FilledCandle(T3), // trailing — removed
        };
        var result = candles.TrimTrailingFilledCandles();
        Assert.Equal(2, result.Count);
        Assert.DoesNotContain(T2, result.Keys);
        Assert.DoesNotContain(T3, result.Keys);
    }

    [Fact]
    public void TrimTrailing_FilledInMiddle_NotRemoved()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.RealCandle(T0),
            [T1] = TimeSeriesFixtures.FilledCandle(T1), // middle — kept
            [T2] = TimeSeriesFixtures.RealCandle(T2),
        };
        var result = candles.TrimTrailingFilledCandles();
        Assert.Equal(3, result.Count);
        Assert.Contains(T1, result.Keys);
    }

    // ── TrimLeadingAndTrailingFilledCandles ───────────────────────────────────

    [Fact]
    public void TrimBoth_NullInput_ThrowsArgumentNullException()
    {
        IReadOnlyDictionary<DateTime, TimeSeriesValue>? candles = null;
        Assert.Throws<ArgumentNullException>(() => candles!.TrimLeadingAndTrailingFilledCandles());
    }

    [Fact]
    public void TrimBoth_AllReal_ReturnsSameCount()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.RealCandle(T0),
            [T1] = TimeSeriesFixtures.RealCandle(T1),
        };
        Assert.Equal(2, candles.TrimLeadingAndTrailingFilledCandles().Count);
    }

    [Fact]
    public void TrimBoth_AllFilled_ReturnsEmpty()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.FilledCandle(T0),
            [T1] = TimeSeriesFixtures.FilledCandle(T1),
        };
        Assert.Empty(candles.TrimLeadingAndTrailingFilledCandles());
    }

    [Fact]
    public void TrimBoth_EmptyDictionary_ReturnsEmpty()
    {
        Assert.Empty(new Dictionary<DateTime, TimeSeriesValue>().TrimLeadingAndTrailingFilledCandles());
    }

    [Fact]
    public void TrimBoth_LeadingAndTrailingFilled_RemovesBothEnds()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.FilledCandle(T0), // leading — removed
            [T1] = TimeSeriesFixtures.RealCandle(T1),
            [T2] = TimeSeriesFixtures.RealCandle(T2),
            [T3] = TimeSeriesFixtures.FilledCandle(T3), // trailing — removed
        };
        var result = candles.TrimLeadingAndTrailingFilledCandles();
        Assert.Equal(2, result.Count);
        Assert.DoesNotContain(T0, result.Keys);
        Assert.DoesNotContain(T3, result.Keys);
    }

    [Fact]
    public void TrimBoth_FilledInMiddleNotRemoved()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.FilledCandle(T0), // leading — removed
            [T1] = TimeSeriesFixtures.RealCandle(T1),
            [T2] = TimeSeriesFixtures.FilledCandle(T2), // middle — kept
            [T3] = TimeSeriesFixtures.RealCandle(T3),
            [T4] = TimeSeriesFixtures.FilledCandle(T4), // trailing — removed
        };
        var result = candles.TrimLeadingAndTrailingFilledCandles();
        Assert.Equal(3, result.Count);
        Assert.Contains(T2, result.Keys);
        Assert.DoesNotContain(T0, result.Keys);
        Assert.DoesNotContain(T4, result.Keys);
    }

    [Fact]
    public void TrimBoth_SingleRealCandle_ReturnsThatCandle()
    {
        var candles = new Dictionary<DateTime, TimeSeriesValue>
        {
            [T0] = TimeSeriesFixtures.RealCandle(T0)
        };
        var result = candles.TrimLeadingAndTrailingFilledCandles();
        Assert.Single(result);
    }
}
