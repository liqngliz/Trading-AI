using Integrations.TwelveData;
using Integrations.Tests.Helpers;

namespace Integrations.Tests;

public class BuildMissingRangesTests
{
    private static readonly DateTime T0 = new DateTime(2024, 1, 1, 0, 0, 0);
    private static readonly DateTime T1 = new DateTime(2024, 1, 1, 4, 0, 0);
    private static readonly DateTime T2 = new DateTime(2024, 1, 1, 8, 0, 0);
    private static readonly DateTime T3 = new DateTime(2024, 1, 1, 12, 0, 0);
    private static readonly DateTime T4 = new DateTime(2024, 1, 1, 16, 0, 0);

    private static SortedDictionary<string, TimeSeriesValue> MakeBucket(
        params DateTime[] present)
    {
        var bucket = new SortedDictionary<string, TimeSeriesValue>(StringComparer.Ordinal);
        foreach (var dt in present)
            bucket[TwelveTimeSeriesParamExtensions.ToStorageKey(dt)] = TimeSeriesFixtures.RealCandle(dt);
        return bucket;
    }

    [Fact]
    public void BuildMissingRanges_AllPresent_ReturnsEmptyList()
    {
        var timestamps = new List<DateTime> { T0, T1, T2 };
        var bucket = MakeBucket(T0, T1, T2);
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(timestamps, bucket);
        Assert.Empty(result);
    }

    [Fact]
    public void BuildMissingRanges_NonePresent_ReturnsSingleRange()
    {
        var timestamps = new List<DateTime> { T0, T1, T2 };
        var bucket = MakeBucket(); // empty cache
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(timestamps, bucket);
        Assert.Single(result);
        Assert.Equal(T0, result[0].Start);
        Assert.Equal(T2, result[0].End);
    }

    [Fact]
    public void BuildMissingRanges_SingleMissingInMiddle_ReturnsSingleRange()
    {
        var timestamps = new List<DateTime> { T0, T1, T2 };
        var bucket = MakeBucket(T0, T2); // T1 missing
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(timestamps, bucket);
        Assert.Single(result);
        Assert.Equal(T1, result[0].Start);
        Assert.Equal(T1, result[0].End);
    }

    [Fact]
    public void BuildMissingRanges_TwoContiguousMissing_ReturnsSingleRange()
    {
        var timestamps = new List<DateTime> { T0, T1, T2, T3 };
        var bucket = MakeBucket(T0, T3); // T1 and T2 missing (contiguous 4h gap)
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(timestamps, bucket);
        Assert.Single(result);
        Assert.Equal(T1, result[0].Start);
        Assert.Equal(T2, result[0].End);
    }

    [Fact]
    public void BuildMissingRanges_TwoSeparateGaps_ReturnsTwoRanges()
    {
        // Algorithm computes step = gap between FIRST two missing items.
        // A new range starts only when a subsequent gap DIFFERS from that step.
        //
        // missing = [T0, T1, T4]: step = T1-T0 = 4h.
        //   T0→T1 gap = 4h == step  → same range
        //   T1→T4 gap = 12h != step → new range
        // Result: [(T0,T1), (T4,T4)]
        var timestamps = new List<DateTime> { T0, T1, T2, T3, T4 };
        var bucket = MakeBucket(T2, T3); // T0, T1, T4 are missing
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(timestamps, bucket);
        Assert.Equal(2, result.Count);
        Assert.Equal(T0, result[0].Start);
        Assert.Equal(T1, result[0].End);
        Assert.Equal(T4, result[1].Start);
        Assert.Equal(T4, result[1].End);
    }

    [Fact]
    public void BuildMissingRanges_MissingAtStart_RangeStartsAtStart()
    {
        var timestamps = new List<DateTime> { T0, T1, T2 };
        var bucket = MakeBucket(T1, T2);
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(timestamps, bucket);
        Assert.Single(result);
        Assert.Equal(T0, result[0].Start);
    }

    [Fact]
    public void BuildMissingRanges_MissingAtEnd_RangeEndsAtEnd()
    {
        var timestamps = new List<DateTime> { T0, T1, T2 };
        var bucket = MakeBucket(T0, T1);
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(timestamps, bucket);
        Assert.Single(result);
        Assert.Equal(T2, result[0].End);
    }

    [Fact]
    public void BuildMissingRanges_OnlyOneMissing_ReturnsSingleRangeWithSameStartEnd()
    {
        // When only one timestamp is missing, step = TimeSpan.Zero, single range with Start==End
        var timestamps = new List<DateTime> { T0, T1, T2 };
        var bucket = MakeBucket(T0, T2);
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(timestamps, bucket);
        Assert.Single(result);
        Assert.Equal(T1, result[0].Start);
        Assert.Equal(T1, result[0].End);
    }

    [Fact]
    public void BuildMissingRanges_EmptyExpected_ReturnsEmpty()
    {
        var result = TwelveTimeSeriesParamExtensions.BuildMissingRanges(
            new List<DateTime>(),
            MakeBucket());
        Assert.Empty(result);
    }
}
