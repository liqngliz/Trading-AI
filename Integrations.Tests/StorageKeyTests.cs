using Integrations.TwelveData;

namespace Integrations.Tests;

public class StorageKeyTests
{
    [Fact]
    public void ToStorageKey_FormatIsYyyyMMddHHmmss()
    {
        var dt = new DateTime(2024, 3, 15, 8, 30, 0, DateTimeKind.Unspecified);
        Assert.Equal("2024-03-15 08:30:00", TwelveTimeSeriesParamExtensions.ToStorageKey(dt));
    }

    [Fact]
    public void ToStorageKey_MidnightDateTime_ProducesCorrectString()
    {
        var dt = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Unspecified);
        Assert.Equal("2024-01-01 00:00:00", TwelveTimeSeriesParamExtensions.ToStorageKey(dt));
    }

    [Fact]
    public void ToStorageKey_NonMidnightDateTime_PreservesHoursMinutes()
    {
        var dt = new DateTime(2024, 6, 20, 16, 0, 0, DateTimeKind.Unspecified);
        Assert.Equal("2024-06-20 16:00:00", TwelveTimeSeriesParamExtensions.ToStorageKey(dt));
    }

    [Fact]
    public void ParseStorageKey_ValidString_ParsesCorrectly()
    {
        var result = TwelveTimeSeriesParamExtensions.ParseStorageKey("2024-03-15 08:30:00");
        Assert.Equal(new DateTime(2024, 3, 15, 8, 30, 0), result);
    }

    [Fact]
    public void ParseStorageKey_RoundTrip_MatchesOriginal()
    {
        var original = new DateTime(2024, 11, 5, 12, 0, 0, DateTimeKind.Unspecified);
        var key = TwelveTimeSeriesParamExtensions.ToStorageKey(original);
        var parsed = TwelveTimeSeriesParamExtensions.ParseStorageKey(key);
        Assert.Equal(original, parsed);
    }

    [Fact]
    public void ParseStorageKey_InvalidFormat_ThrowsFormatException()
    {
        Assert.Throws<FormatException>(() =>
            TwelveTimeSeriesParamExtensions.ParseStorageKey("2024/01/01"));
    }

    [Fact]
    public void ToStorageKey_ThenParseStorageKey_SecondsPreserved()
    {
        var original = new DateTime(2024, 1, 1, 4, 0, 0, DateTimeKind.Unspecified);
        var roundTripped = TwelveTimeSeriesParamExtensions.ParseStorageKey(
            TwelveTimeSeriesParamExtensions.ToStorageKey(original));
        Assert.Equal(0, roundTripped.Second);
        Assert.Equal(original, roundTripped);
    }
}
