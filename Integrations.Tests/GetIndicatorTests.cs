using System.Net;
using Integrations.TwelveData;
using Integrations.Tests.Helpers;
using Moq;

namespace Integrations.Tests;

public class GetIndicatorTests
{
    private static readonly DateTime Start = new DateTime(2024, 1, 1, 0, 0, 0);
    private static readonly DateTime End = new DateTime(2024, 1, 2, 0, 0, 0);

    private static (TwelveDataParam param, FakeHttpMessageHandler handler) BuildIndicatorParam(
        TwelveDataEndpoint endpoint,
        IRepository<IndicatorCacheDocument>? indicatorRepository = null,
        DateTime? startDate = null,
        DateTime? endDate = null)
    {
        var handler = new FakeHttpMessageHandler();
        var httpClient = new HttpClient(handler)
        {
            BaseAddress = new Uri("https://api.twelvedata.com/")
        };

        var param = new TwelveDataParam(
            httpClient: httpClient,
            repository: new Mock<IRepository<TimeSeriesCacheDocument>>().Object,
            apiKey: "test-key",
            symbol: "AAPL",
            startDate: startDate ?? Start,
            endDate: endDate ?? End,
            format: TwelveDataFormat.Json,
            endpoint: endpoint,
            indicatorRepository: indicatorRepository);

        return (param, handler);
    }

    private static Mock<IRepository<IndicatorCacheDocument>> EmptyIndicatorRepo()
    {
        var mock = new Mock<IRepository<IndicatorCacheDocument>>();
        mock.Setup(r => r.GetAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync((IndicatorCacheDocument?)null);
        mock.Setup(r => r.SaveAsync(It.IsAny<string>(), It.IsAny<IndicatorCacheDocument>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);
        return mock;
    }

    private static Mock<IRepository<IndicatorCacheDocument>> PreloadedIndicatorRepo(
        TwelveDataEndpoint endpoint, string interval, params (DateTime dt, decimal value)[] entries)
    {
        var bucketKey = $"{endpoint.ToString().ToLowerInvariant()}/{interval}";
        var bucket = new SortedDictionary<string, IndicatorValue>(StringComparer.Ordinal);
        foreach (var (dt, value) in entries)
            bucket[TwelveDataParamExtensions.ToStorageKey(dt)] = new IndicatorValue(dt, value, IsFilled: false);

        var doc = new IndicatorCacheDocument
        {
            Symbol = "AAPL",
            Data = new Dictionary<string, SortedDictionary<string, IndicatorValue>> { [bucketKey] = bucket }
        };

        var mock = new Mock<IRepository<IndicatorCacheDocument>>();
        mock.Setup(r => r.GetAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(doc);
        mock.Setup(r => r.SaveAsync(It.IsAny<string>(), It.IsAny<IndicatorCacheDocument>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);
        return mock;
    }

    // Builds {"values":[{"datetime":"...","<fieldName>":"..."},...]}
    private static string BuildIndicatorJsonPayload(string fieldName, params (string datetime, string value)[] rows)
    {
        var items = string.Join(",", rows.Select(r =>
            $"{{\"datetime\":\"{r.datetime}\",\"{fieldName}\":\"{r.value}\"}}"));
        return $"{{\"values\":[{items}]}}";
    }

    // --- Guard tests ---

    [Fact]
    public async Task GetIndicator_NullParam_ThrowsArgumentNullException()
    {
        TwelveDataParam? param = null;
        await Assert.ThrowsAsync<ArgumentNullException>(() => TwelveDataIndicator.GetIndicator(param!));
    }

    [Fact]
    public async Task GetIndicator_TimeSeriesEndpoint_ThrowsInvalidOperationException()
    {
        var (param, _) = BuildIndicatorParam(TwelveDataEndpoint.TimeSeries);
        await Assert.ThrowsAsync<InvalidOperationException>(() => TwelveDataIndicator.GetIndicator(param));
    }

    // --- Happy path for each endpoint ---

    [Fact]
    public async Task GetIndicator_Obv_ReturnsCorrectValue()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("obv", ("2024-01-01 00:00:00", "123456.78")));

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Single(result);
        Assert.Equal(123456.78m, result[new DateTime(2024, 1, 1, 0, 0, 0)].Value);
        Assert.False(result[new DateTime(2024, 1, 1, 0, 0, 0)].IsFilled);
    }

    [Fact]
    public async Task GetIndicator_Ad_ReturnsCorrectValue()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Ad);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("ad", ("2024-01-01 00:00:00", "9876.54")));

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Single(result);
        Assert.Equal(9876.54m, result[new DateTime(2024, 1, 1, 0, 0, 0)].Value);
    }

    [Fact]
    public async Task GetIndicator_Adosc_ReturnsCorrectValue()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Adosc);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("adosc", ("2024-01-01 00:00:00", "500.00")));

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Single(result);
        Assert.Equal(500.00m, result[new DateTime(2024, 1, 1, 0, 0, 0)].Value);
    }

    [Fact]
    public async Task GetIndicator_Rvol_ReturnsCorrectValue()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Rvol);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("rvol", ("2024-01-01 00:00:00", "1.23")));

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Single(result);
        Assert.Equal(1.23m, result[new DateTime(2024, 1, 1, 0, 0, 0)].Value);
    }

    [Fact]
    public async Task GetIndicator_MultipleRows_ReturnsAll()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("obv",
                ("2024-01-01 00:00:00", "100.0"),
                ("2024-01-01 04:00:00", "200.0"),
                ("2024-01-01 08:00:00", "300.0")));

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Equal(3, result.Count);
        Assert.Equal(100.0m, result[new DateTime(2024, 1, 1, 0, 0, 0)].Value);
        Assert.Equal(200.0m, result[new DateTime(2024, 1, 1, 4, 0, 0)].Value);
        Assert.Equal(300.0m, result[new DateTime(2024, 1, 1, 8, 0, 0)].Value);
    }

    // --- No-data / error response ---

    [Fact]
    public async Task GetIndicator_NoDataResponse_ReturnsEmptyDictionary()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv);
        handler.EnqueueResponse(HttpStatusCode.OK, TimeSeriesFixtures.BuildNoDataJson());

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Empty(result);
    }

    [Fact]
    public async Task GetIndicator_ApiReturns500_ThrowsHttpRequestException()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv);
        handler.EnqueueResponse(HttpStatusCode.InternalServerError, "Internal Server Error");

        await Assert.ThrowsAsync<HttpRequestException>(() => TwelveDataIndicator.GetIndicator(param));
    }

    // --- URL construction ---

    [Fact]
    public async Task GetIndicator_ObvEndpoint_RequestUriContainsObvPath()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("obv", ("2024-01-01 00:00:00", "1.0")));

        await TwelveDataIndicator.GetIndicator(param);

        Assert.Contains("/obv", handler.SentRequests[0].RequestUri!.ToString());
    }

    [Fact]
    public async Task GetIndicator_AdEndpoint_RequestUriContainsAdPath()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Ad);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("ad", ("2024-01-01 00:00:00", "1.0")));

        await TwelveDataIndicator.GetIndicator(param);

        Assert.Contains("/ad", handler.SentRequests[0].RequestUri!.ToString());
    }

    [Fact]
    public async Task GetIndicator_RequestUri_ContainsSymbolApiKeyAndInterval()
    {
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("obv", ("2024-01-01 00:00:00", "1.0")));

        await TwelveDataIndicator.GetIndicator(param);

        var uri = handler.SentRequests[0].RequestUri!.ToString();
        Assert.Contains("symbol=AAPL", uri);
        Assert.Contains("apikey=test-key", uri);
        Assert.Contains("interval=4h", uri);
    }

    // --- Caching ---

    [Fact]
    public async Task GetIndicator_WithEmptyCache_CallsApiAndSaves()
    {
        var repoMock = EmptyIndicatorRepo();
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv, repoMock.Object);
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("obv", ("2024-01-01 00:00:00", "999.0")));

        await TwelveDataIndicator.GetIndicator(param);

        Assert.NotEmpty(handler.SentRequests);
        repoMock.Verify(r => r.SaveAsync(
            It.IsAny<string>(),
            It.IsAny<IndicatorCacheDocument>(),
            It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task GetIndicator_FullyCached_DoesNotCallApi()
    {
        // Start..Start+8h with 4h interval produces exactly 1 expected timestamp (Start),
        // which is already in the cache — no API call should be made.
        var narrowEnd = Start.AddHours(8);
        var repoMock = PreloadedIndicatorRepo(TwelveDataEndpoint.Obv, "4h", (Start, 500m));
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv, repoMock.Object, endDate: narrowEnd);

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Empty(handler.SentRequests);
        Assert.Equal(500m, result[Start].Value);
    }

    [Fact]
    public async Task GetIndicator_FullyCached_SaveAsyncNotCalled()
    {
        var narrowEnd = Start.AddHours(8);
        var repoMock = PreloadedIndicatorRepo(TwelveDataEndpoint.Obv, "4h", (Start, 500m));
        var (param, _) = BuildIndicatorParam(TwelveDataEndpoint.Obv, repoMock.Object, endDate: narrowEnd);

        await TwelveDataIndicator.GetIndicator(param);

        repoMock.Verify(r => r.SaveAsync(
            It.IsAny<string>(),
            It.IsAny<IndicatorCacheDocument>(),
            It.IsAny<CancellationToken>()), Times.Never);
    }

    [Fact]
    public async Task GetIndicator_NullIndicatorRepository_CallsApiWithoutCaching()
    {
        // No indicatorRepository — falls back to direct API call, no save
        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv); // null repo
        handler.EnqueueResponse(HttpStatusCode.OK,
            BuildIndicatorJsonPayload("obv", ("2024-01-01 00:00:00", "1.0")));

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Single(handler.SentRequests);
        Assert.Equal(1.0m, result[Start].Value);
    }

    [Fact]
    public async Task GetIndicator_ResultFilteredToRequestedRange()
    {
        // Narrow range: Start..Start+8h produces 1 expected timestamp (Start).
        // Cache is wider (also has wideStart from the day before) — that entry must be excluded.
        var narrowEnd = Start.AddHours(8);
        var wideStart = new DateTime(2023, 12, 31, 0, 0, 0);
        var repoMock = PreloadedIndicatorRepo(TwelveDataEndpoint.Obv, "4h",
            (wideStart, 100m),  // outside requested range
            (Start, 200m));     // inside requested range

        var (param, handler) = BuildIndicatorParam(TwelveDataEndpoint.Obv, repoMock.Object, endDate: narrowEnd);

        var result = await TwelveDataIndicator.GetIndicator(param);

        Assert.Empty(handler.SentRequests);
        Assert.DoesNotContain(wideStart, result.Keys);
        Assert.Equal(200m, result[Start].Value);
    }

    // --- ParseIndicatorJson unit tests (internal) ---

    [Fact]
    public void ParseIndicatorJson_SingleRow_ParsesCorrectly()
    {
        var json = BuildIndicatorJsonPayload("obv", ("2024-01-01 00:00:00", "42.5"));

        var result = TwelveDataIndicator.ParseIndicatorJson(json, "obv");

        Assert.Single(result);
        Assert.Equal(42.5m, result[new DateTime(2024, 1, 1, 0, 0, 0)]);
    }

    [Fact]
    public void ParseIndicatorJson_MultipleRows_ParsesAll()
    {
        var json = BuildIndicatorJsonPayload("ad",
            ("2024-01-01 00:00:00", "10.0"),
            ("2024-01-01 04:00:00", "20.0"));

        var result = TwelveDataIndicator.ParseIndicatorJson(json, "ad");

        Assert.Equal(2, result.Count);
        Assert.Equal(10.0m, result[new DateTime(2024, 1, 1, 0, 0, 0)]);
        Assert.Equal(20.0m, result[new DateTime(2024, 1, 1, 4, 0, 0)]);
    }

    [Fact]
    public void ParseIndicatorJson_EmptyValuesArray_ReturnsEmptyDictionary()
    {
        var result = TwelveDataIndicator.ParseIndicatorJson("{\"values\":[]}", "obv");

        Assert.Empty(result);
    }

    [Fact]
    public void ParseIndicatorJson_NoDataResponse_ReturnsEmptyDictionary()
    {
        var result = TwelveDataIndicator.ParseIndicatorJson(
            TimeSeriesFixtures.BuildNoDataJson(), "obv");

        Assert.Empty(result);
    }

    [Fact]
    public void ParseIndicatorJson_MissingValuesArray_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() =>
            TwelveDataIndicator.ParseIndicatorJson("{\"status\":\"ok\"}", "obv"));
    }

    [Fact]
    public void ParseIndicatorJson_WrongFieldName_ThrowsKeyNotFoundException()
    {
        // JSON has "obv" field but we ask for "ad"
        var json = BuildIndicatorJsonPayload("obv", ("2024-01-01 00:00:00", "100.0"));

        Assert.Throws<KeyNotFoundException>(() =>
            TwelveDataIndicator.ParseIndicatorJson(json, "ad"));
    }
}
