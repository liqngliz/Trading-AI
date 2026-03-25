using Integrations.TwelveData;
using Integrations.Tests.Helpers;

namespace Integrations.Tests;

public class ParseJsonTests
{
    [Fact]
    public void ParseJson_ValidSingleCandle_ReturnsSingleEntry()
    {
        var json = TimeSeriesFixtures.BuildJsonPayload(("2024-01-02 00:00:00", "1850.50", "1870.00", "1840.25", "1865.75"));
        var result = TwelveDataSeries.ParseJson(json);
        Assert.Single(result);
    }

    [Fact]
    public void ParseJson_ValidMultipleCandles_ReturnsAllEntries()
    {
        var json = TimeSeriesFixtures.BuildJsonPayload(
            ("2024-01-01 00:00:00", "1900", "1950", "1880", "1920"),
            ("2024-01-01 04:00:00", "1920", "1960", "1910", "1945"),
            ("2024-01-01 08:00:00", "1945", "1970", "1935", "1960"));
        var result = TwelveDataSeries.ParseJson(json);
        Assert.Equal(3, result.Count);
    }

    [Fact]
    public void ParseJson_CandleValues_OpenHighLowCloseAreCorrect()
    {
        var json = TimeSeriesFixtures.BuildJsonPayload(("2024-01-02 00:00:00", "1850.50", "1870.00", "1840.25", "1865.75"));
        var result = TwelveDataSeries.ParseJson(json);
        var candle = result.Values.Single();
        Assert.Equal(1850.50m, candle.Open);
        Assert.Equal(1870.00m, candle.High);
        Assert.Equal(1840.25m, candle.Low);
        Assert.Equal(1865.75m, candle.Close);
    }

    [Fact]
    public void ParseJson_ParsedCandle_IsFilled_IsFalse()
    {
        var json = TimeSeriesFixtures.BuildJsonPayload(("2024-01-02 00:00:00", "1900", "1950", "1880", "1920"));
        var result = TwelveDataSeries.ParseJson(json);
        Assert.False(result.Values.Single().IsFilled);
    }

    [Fact]
    public void ParseJson_ParsedCandle_Datetime_ParsedCorrectly()
    {
        var json = TimeSeriesFixtures.BuildJsonPayload(("2024-01-02 08:00:00", "1900", "1950", "1880", "1920"));
        var result = TwelveDataSeries.ParseJson(json);
        var key = result.Keys.Single();
        Assert.Equal(new DateTime(2024, 1, 2, 8, 0, 0), key);
    }

    [Fact]
    public void ParseJson_NoDataErrorResponse_ReturnsEmptyDictionary()
    {
        var json = TimeSeriesFixtures.BuildNoDataJson();
        var result = TwelveDataSeries.ParseJson(json);
        Assert.Empty(result);
    }

    [Fact]
    public void ParseJson_EmptyValuesArray_ReturnsEmptyDictionary()
    {
        var json = """{"values":[]}""";
        var result = TwelveDataSeries.ParseJson(json);
        Assert.Empty(result);
    }

    [Fact]
    public void ParseJson_MissingValuesArray_ThrowsInvalidOperationException()
    {
        var json = """{"status":"ok"}""";
        Assert.Throws<InvalidOperationException>(() =>
            TwelveDataSeries.ParseJson(json));
    }

    [Fact]
    public void ParseJson_ErrorStatusWithCode200_ThrowsInvalidOperationException()
    {
        // Unrecognised error codes (not 400/429/404) throw regardless of values array
        var json = """{"status":"error","code":200,"message":"Something else","values":[]}""";
        Assert.Throws<InvalidOperationException>(() =>
            TwelveDataSeries.ParseJson(json));
    }

    [Fact]
    public void ParseJson_ErrorCode400AnyMessage_ReturnsEmpty()
    {
        // All code=400 responses are treated as "no data" — message is not checked
        var json = """{"status":"error","code":400,"message":"Invalid API key."}""";
        var result = TwelveDataSeries.ParseJson(json);
        Assert.Empty(result);
    }
}
