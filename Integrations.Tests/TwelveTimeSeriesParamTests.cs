using Integrations.TwelveData;
using Moq;

namespace Integrations.Tests;

public class TwelveTimeSeriesParamTests
{
    private static HttpClient ValidHttpClient() =>
        new HttpClient { BaseAddress = new Uri("https://api.twelvedata.com/") };

    private static IRepository<TimeSeriesCacheDocument> ValidRepo() =>
        new Mock<IRepository<TimeSeriesCacheDocument>>().Object;

    private static DateTime Start => new DateTime(2024, 1, 1);
    private static DateTime End => new DateTime(2024, 1, 2);

    [Fact]
    public void Constructor_NullHttpClient_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new TwelveTimeSeriesParam(null!, ValidRepo(), "key", "AAPL", Start, End, TwelveDataFormat.Json));
    }

    [Fact]
    public void Constructor_NullBaseAddress_ThrowsInvalidOperationException()
    {
        var client = new HttpClient(); // no BaseAddress
        Assert.Throws<InvalidOperationException>(() =>
            new TwelveTimeSeriesParam(client, ValidRepo(), "key", "AAPL", Start, End, TwelveDataFormat.Json));
    }

    [Fact]
    public void Constructor_NullRepository_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new TwelveTimeSeriesParam(ValidHttpClient(), null!, "key", "AAPL", Start, End, TwelveDataFormat.Json));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_InvalidApiKey_ThrowsArgumentException(string? apiKey)
    {
        Assert.Throws<ArgumentException>(() =>
            new TwelveTimeSeriesParam(ValidHttpClient(), ValidRepo(), apiKey!, "AAPL", Start, End, TwelveDataFormat.Json));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_InvalidSymbol_ThrowsArgumentException(string? symbol)
    {
        Assert.Throws<ArgumentException>(() =>
            new TwelveTimeSeriesParam(ValidHttpClient(), ValidRepo(), "key", symbol!, Start, End, TwelveDataFormat.Json));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_InvalidInterval_ThrowsArgumentException(string? interval)
    {
        Assert.Throws<ArgumentException>(() =>
            new TwelveTimeSeriesParam(ValidHttpClient(), ValidRepo(), "key", "AAPL", Start, End, TwelveDataFormat.Json, interval: interval!));
    }

    [Fact]
    public void Constructor_StartDateAfterEndDate_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new TwelveTimeSeriesParam(ValidHttpClient(), ValidRepo(), "key", "AAPL", End, Start, TwelveDataFormat.Json));
    }

    [Fact]
    public void Constructor_StartDateEqualsEndDate_Succeeds()
    {
        // Boundary: == is allowed; only strict > throws
        var param = new TwelveTimeSeriesParam(ValidHttpClient(), ValidRepo(), "key", "AAPL", Start, Start, TwelveDataFormat.Json);
        Assert.Equal(Start, param.StartDate);
        Assert.Equal(Start, param.EndDate);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_NonPositiveOutputSize_ThrowsArgumentOutOfRangeException(int outputSize)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TwelveTimeSeriesParam(ValidHttpClient(), ValidRepo(), "key", "AAPL", Start, End, TwelveDataFormat.Json, outputSize: outputSize));
    }

    [Fact]
    public void Constructor_ValidArguments_PropertiesSetCorrectly()
    {
        var client = ValidHttpClient();
        var repo = ValidRepo();
        var param = new TwelveTimeSeriesParam(client, repo, "mykey", "XAU/USD", Start, End, TwelveDataFormat.Csv, "1day", 1000);

        Assert.Same(client, param.HttpClient);
        Assert.Same(repo, param.Repository);
        Assert.Equal("mykey", param.ApiKey);
        Assert.Equal("XAU/USD", param.Symbol);
        Assert.Equal(Start, param.StartDate);
        Assert.Equal(End, param.EndDate);
        Assert.Equal(TwelveDataFormat.Csv, param.Format);
        Assert.Equal("1day", param.Interval);
        Assert.Equal(1000, param.OutputSize);
    }

    [Fact]
    public void Constructor_DefaultInterval_Is4h()
    {
        var param = new TwelveTimeSeriesParam(ValidHttpClient(), ValidRepo(), "key", "AAPL", Start, End, TwelveDataFormat.Json);
        Assert.Equal("4h", param.Interval);
    }

    [Fact]
    public void Constructor_DefaultOutputSize_Is5000()
    {
        var param = new TwelveTimeSeriesParam(ValidHttpClient(), ValidRepo(), "key", "AAPL", Start, End, TwelveDataFormat.Json);
        Assert.Equal(5000, param.OutputSize);
    }
}
