using Integrations.TwelveData;
using Integrations.Tests.Helpers;

namespace Integrations.Tests;

public class ParseCsvTests
{
    [Fact]
    public void ParseCsv_ValidSingleDataRow_ReturnsSingleEntry()
    {
        var csv = TimeSeriesFixtures.BuildCsvPayload(("2024-01-02 00:00:00", "1850.50", "1870.00", "1840.25", "1865.75"));
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        Assert.Single(result);
    }

    [Fact]
    public void ParseCsv_ValidMultipleDataRows_ReturnsAllEntries()
    {
        var csv = TimeSeriesFixtures.BuildCsvPayload(
            ("2024-01-01 00:00:00", "1900", "1950", "1880", "1920"),
            ("2024-01-01 04:00:00", "1920", "1960", "1910", "1945"),
            ("2024-01-01 08:00:00", "1945", "1970", "1935", "1960"));
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        Assert.Equal(3, result.Count);
    }

    [Fact]
    public void ParseCsv_CandleValues_AllFieldsCorrect()
    {
        var csv = TimeSeriesFixtures.BuildCsvPayload(("2024-01-02 00:00:00", "1850.50", "1870.00", "1840.25", "1865.75"));
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        var candle = result.Values.Single();
        Assert.Equal(1850.50m, candle.Open);
        Assert.Equal(1870.00m, candle.High);
        Assert.Equal(1840.25m, candle.Low);
        Assert.Equal(1865.75m, candle.Close);
    }

    [Fact]
    public void ParseCsv_ParsedCandle_IsFilled_IsFalse()
    {
        var csv = TimeSeriesFixtures.BuildCsvPayload(("2024-01-02 00:00:00", "1900", "1950", "1880", "1920"));
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        Assert.False(result.Values.Single().IsFilled);
    }

    [Fact]
    public void ParseCsv_ParsedCandle_Datetime_ParsedCorrectly()
    {
        var csv = TimeSeriesFixtures.BuildCsvPayload(("2024-01-02 08:00:00", "1900", "1950", "1880", "1920"));
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        Assert.Equal(new DateTime(2024, 1, 2, 8, 0, 0), result.Keys.Single());
    }

    [Fact]
    public void ParseCsv_HeaderOnlyNoDataRows_ReturnsEmptyDictionary()
    {
        var csv = "datetime,open,high,low,close\n";
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        Assert.Empty(result);
    }

    [Fact]
    public void ParseCsv_EmptyString_ReturnsEmptyDictionary()
    {
        var result = TwelveTimeSeriesParamExtensions.ParseCsv("");
        Assert.Empty(result);
    }

    [Fact]
    public void ParseCsv_WindowsLineEndings_ParsedCorrectly()
    {
        var csv = "datetime,open,high,low,close\r\n2024-01-02 00:00:00,1900,1950,1880,1920\r\n";
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        Assert.Single(result);
    }

    [Fact]
    public void ParseCsv_UnixLineEndings_ParsedCorrectly()
    {
        var csv = "datetime,open,high,low,close\n2024-01-02 00:00:00,1900,1950,1880,1920\n";
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        Assert.Single(result);
    }

    [Fact]
    public void ParseCsv_RowWithFewerThan5Parts_IsSkipped()
    {
        // Row with only 3 fields — should be silently skipped
        var csv = "datetime,open,high,low,close\n2024-01-02 00:00:00,1900,1950\n2024-01-03 00:00:00,1900,1950,1880,1920\n";
        var result = TwelveTimeSeriesParamExtensions.ParseCsv(csv);
        Assert.Single(result);
    }
}
