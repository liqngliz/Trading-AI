using Integrations.TwelveData;

namespace Integrations.Tests;

public class BuildExpectedTimestampsTests
{
    private static readonly DateTime Base = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Unspecified);

    [Fact]
    public void BuildExpectedTimestamps_4hInterval_FirstTimestampIsStartDate()
    {
        var result = TwelveTimeSeriesParamExtensions.BuildExpectedTimestamps(Base, Base.AddDays(2), "4h");
        Assert.Equal(Base, result[0]);
    }

    [Fact]
    public void BuildExpectedTimestamps_4hInterval_StepsAre4Hours()
    {
        var result = TwelveTimeSeriesParamExtensions.BuildExpectedTimestamps(Base, Base.AddDays(1), "4h");
        Assert.True(result.Count >= 2);
        Assert.Equal(TimeSpan.FromHours(4), result[1] - result[0]);
    }

    [Fact]
    public void BuildExpectedTimestamps_EndOnIntervalBoundary_ExcludesLastIncompleteStep()
    {
        // start=00:00, end=08:00 with 4h step.
        // At 04:00: next=08:00 >= end=08:00 AND 04:00 != 08:00 → break before adding 04:00.
        // Only 00:00 is included (next=04:00 < 08:00 passes the check).
        var start = Base;
        var end = Base.AddHours(8);
        var result = TwelveTimeSeriesParamExtensions.BuildExpectedTimestamps(start, end, "4h");
        Assert.Single(result);
        Assert.Equal(Base, result[0]);
    }

    [Fact]
    public void BuildExpectedTimestamps_EndBetweenIntervals_ExcludesPartialStep()
    {
        // start=00:00, end=05:30 with 4h step.
        // At 00:00: next=04:00 < 05:30 → include 00:00.
        // At 04:00: next=08:00 >= 05:30 AND 04:00 != 05:30 → break.
        var start = Base;
        var end = Base.AddHours(5).AddMinutes(30);
        var result = TwelveTimeSeriesParamExtensions.BuildExpectedTimestamps(start, end, "4h");
        Assert.Single(result);
        Assert.Equal(Base, result[0]);
    }

    [Fact]
    public void BuildExpectedTimestamps_StartEqualsEnd_ReturnsSingleEntry()
    {
        // When start == end: current.Add(step) >= end is true, but (current != end) is FALSE,
        // so the break condition evaluates to false and start is added.
        var result = TwelveTimeSeriesParamExtensions.BuildExpectedTimestamps(Base, Base, "4h");
        Assert.Single(result);
        Assert.Equal(Base, result[0]);
    }

    [Fact]
    public void BuildExpectedTimestamps_1dayInterval_CorrectCount()
    {
        // start=Jan1, end=Jan8 (7-day span).
        // Jan7: next=Jan8 >= Jan8 AND Jan7 != Jan8 → break. Jan6 is the last included.
        // Produces Jan1..Jan6 = 6 timestamps.
        var start = Base;
        var end = Base.AddDays(7);
        var result = TwelveTimeSeriesParamExtensions.BuildExpectedTimestamps(start, end, "1day");
        Assert.Equal(6, result.Count);
    }

    [Fact]
    public void BuildExpectedTimestamps_RangeSmallerThanOneStep_ReturnsEmpty()
    {
        // end < start + step and end != start: break fires immediately before adding start.
        var start = Base;
        var end = Base.AddHours(3).AddMinutes(59);
        var result = TwelveTimeSeriesParamExtensions.BuildExpectedTimestamps(start, end, "4h");
        Assert.Empty(result);
    }
}
