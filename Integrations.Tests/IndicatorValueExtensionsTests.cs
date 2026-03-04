using Integrations.TwelveData;

namespace Integrations.Tests;

public class IndicatorValueExtensionsTests
{
    private static readonly DateTime T0 = new DateTime(2024, 1, 1, 0, 0, 0);
    private static readonly DateTime T1 = new DateTime(2024, 1, 1, 4, 0, 0);
    private static readonly DateTime T2 = new DateTime(2024, 1, 1, 8, 0, 0);
    private static readonly DateTime T3 = new DateTime(2024, 1, 1, 12, 0, 0);
    private static readonly DateTime T4 = new DateTime(2024, 1, 1, 16, 0, 0);

    // ── TrimLeadingZeroValues ─────────────────────────────────────────────────

    [Fact]
    public void TrimLeading_NullInput_ThrowsArgumentNullException()
    {
        IReadOnlyDictionary<DateTime, decimal>? values = null;
        Assert.Throws<ArgumentNullException>(() => values!.TrimLeadingZeroValues());
    }

    [Fact]
    public void TrimLeading_AllNonZero_ReturnsSameCount()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 1m, [T1] = 2m };
        Assert.Equal(2, values.TrimLeadingZeroValues().Count);
    }

    [Fact]
    public void TrimLeading_AllZero_ReturnsEmpty()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 0m, [T1] = 0m };
        Assert.Empty(values.TrimLeadingZeroValues());
    }

    [Fact]
    public void TrimLeading_EmptyDictionary_ReturnsEmpty()
    {
        Assert.Empty(new Dictionary<DateTime, decimal>().TrimLeadingZeroValues());
    }

    [Fact]
    public void TrimLeading_LeadingZerosThenNonZero_RemovesLeadingZeros()
    {
        var values = new Dictionary<DateTime, decimal>
        {
            [T0] = 0m, // leading — removed
            [T1] = 0m, // leading — removed
            [T2] = 5m,
            [T3] = 6m,
        };
        var result = values.TrimLeadingZeroValues();
        Assert.Equal(2, result.Count);
        Assert.DoesNotContain(T0, result.Keys);
        Assert.DoesNotContain(T1, result.Keys);
    }

    [Fact]
    public void TrimLeading_ZeroInMiddle_NotRemoved()
    {
        var values = new Dictionary<DateTime, decimal>
        {
            [T0] = 1m,
            [T1] = 0m, // middle zero — kept
            [T2] = 3m,
        };
        var result = values.TrimLeadingZeroValues();
        Assert.Equal(3, result.Count);
        Assert.Contains(T1, result.Keys);
    }

    [Fact]
    public void TrimLeading_SingleZero_ReturnsEmpty()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 0m };
        Assert.Empty(values.TrimLeadingZeroValues());
    }

    [Fact]
    public void TrimLeading_SingleNonZero_ReturnsThatEntry()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 42m };
        var result = values.TrimLeadingZeroValues();
        Assert.Single(result);
        Assert.Equal(42m, result[T0]);
    }

    // ── TrimTrailingZeroValues ────────────────────────────────────────────────

    [Fact]
    public void TrimTrailing_NullInput_ThrowsArgumentNullException()
    {
        IReadOnlyDictionary<DateTime, decimal>? values = null;
        Assert.Throws<ArgumentNullException>(() => values!.TrimTrailingZeroValues());
    }

    [Fact]
    public void TrimTrailing_AllNonZero_ReturnsSameCount()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 1m, [T1] = 2m };
        Assert.Equal(2, values.TrimTrailingZeroValues().Count);
    }

    [Fact]
    public void TrimTrailing_AllZero_ReturnsEmpty()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 0m, [T1] = 0m };
        Assert.Empty(values.TrimTrailingZeroValues());
    }

    [Fact]
    public void TrimTrailing_EmptyDictionary_ReturnsEmpty()
    {
        Assert.Empty(new Dictionary<DateTime, decimal>().TrimTrailingZeroValues());
    }

    [Fact]
    public void TrimTrailing_TrailingZerosRemoved()
    {
        var values = new Dictionary<DateTime, decimal>
        {
            [T0] = 1m,
            [T1] = 2m,
            [T2] = 0m, // trailing — removed
            [T3] = 0m, // trailing — removed
        };
        var result = values.TrimTrailingZeroValues();
        Assert.Equal(2, result.Count);
        Assert.DoesNotContain(T2, result.Keys);
        Assert.DoesNotContain(T3, result.Keys);
    }

    [Fact]
    public void TrimTrailing_ZeroInMiddle_NotRemoved()
    {
        var values = new Dictionary<DateTime, decimal>
        {
            [T0] = 1m,
            [T1] = 0m, // middle zero — kept
            [T2] = 3m,
        };
        var result = values.TrimTrailingZeroValues();
        Assert.Equal(3, result.Count);
        Assert.Contains(T1, result.Keys);
    }

    // ── TrimLeadingAndTrailingZeroValues ──────────────────────────────────────

    [Fact]
    public void TrimBoth_NullInput_ThrowsArgumentNullException()
    {
        IReadOnlyDictionary<DateTime, decimal>? values = null;
        Assert.Throws<ArgumentNullException>(() => values!.TrimLeadingAndTrailingZeroValues());
    }

    [Fact]
    public void TrimBoth_AllNonZero_ReturnsSameCount()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 1m, [T1] = 2m };
        Assert.Equal(2, values.TrimLeadingAndTrailingZeroValues().Count);
    }

    [Fact]
    public void TrimBoth_AllZero_ReturnsEmpty()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 0m, [T1] = 0m };
        Assert.Empty(values.TrimLeadingAndTrailingZeroValues());
    }

    [Fact]
    public void TrimBoth_EmptyDictionary_ReturnsEmpty()
    {
        Assert.Empty(new Dictionary<DateTime, decimal>().TrimLeadingAndTrailingZeroValues());
    }

    [Fact]
    public void TrimBoth_LeadingAndTrailingZeros_RemovesBothEnds()
    {
        var values = new Dictionary<DateTime, decimal>
        {
            [T0] = 0m, // leading — removed
            [T1] = 5m,
            [T2] = 6m,
            [T3] = 0m, // trailing — removed
        };
        var result = values.TrimLeadingAndTrailingZeroValues();
        Assert.Equal(2, result.Count);
        Assert.DoesNotContain(T0, result.Keys);
        Assert.DoesNotContain(T3, result.Keys);
    }

    [Fact]
    public void TrimBoth_ZeroInMiddleNotRemoved()
    {
        var values = new Dictionary<DateTime, decimal>
        {
            [T0] = 0m, // leading — removed
            [T1] = 5m,
            [T2] = 0m, // middle zero — kept
            [T3] = 7m,
            [T4] = 0m, // trailing — removed
        };
        var result = values.TrimLeadingAndTrailingZeroValues();
        Assert.Equal(3, result.Count);
        Assert.Contains(T2, result.Keys);
        Assert.DoesNotContain(T0, result.Keys);
        Assert.DoesNotContain(T4, result.Keys);
    }

    [Fact]
    public void TrimBoth_SingleNonZero_ReturnsThatEntry()
    {
        var values = new Dictionary<DateTime, decimal> { [T0] = 99m };
        var result = values.TrimLeadingAndTrailingZeroValues();
        Assert.Single(result);
        Assert.Equal(99m, result[T0]);
    }
}
