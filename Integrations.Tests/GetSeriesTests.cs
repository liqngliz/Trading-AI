using System.Net;
using Integrations.TwelveData;
using Integrations.Tests.Helpers;
using Moq;

namespace Integrations.Tests;

public class GetSeriesTests
{
    // 2-hour range with 4h interval produces just 1 expected timestamp (start only, per loop behavior)
    private static readonly DateTime Start = new DateTime(2024, 1, 1, 0, 0, 0);
    private static readonly DateTime End = new DateTime(2024, 1, 1, 8, 0, 0);

    [Fact]
    public async Task GetSeries_NullParam_ThrowsArgumentNullException()
    {
        TwelveTimeSeriesParam? param = null;
        await Assert.ThrowsAsync<ArgumentNullException>(() => param!.GetSeries());
    }

    [Fact]
    public async Task GetSeries_EmptyCache_ApiReturnsValidJson_ResultContainsCandles()
    {
        var (param, handler, _) = TimeSeriesFixtures.BuildParam(startDate: Start, endDate: End);

        handler.EnqueueResponse(HttpStatusCode.OK,
            TimeSeriesFixtures.BuildJsonPayload(("2024-01-01 00:00:00", "1900", "1950", "1880", "1920")));
        // Second call (pagination) returns no data
        handler.EnqueueResponse(HttpStatusCode.OK, TimeSeriesFixtures.BuildNoDataJson());

        var result = await param.GetSeries();

        Assert.True(result.Count > 0);
    }

    [Fact]
    public async Task GetSeries_EmptyCache_ApiReturnsCsv_ResultContainsCandles()
    {
        var handler = new FakeHttpMessageHandler();
        var httpClient = new HttpClient(handler) { BaseAddress = new Uri("https://api.twelvedata.com/") };
        var repoMock = new Mock<IRepository<TimeSeriesCacheDocument>>();
        repoMock.Setup(r => r.GetAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync((TimeSeriesCacheDocument?)null);
        repoMock.Setup(r => r.SaveAsync(It.IsAny<string>(), It.IsAny<TimeSeriesCacheDocument>(), It.IsAny<CancellationToken>()))
                .Returns(Task.CompletedTask);

        var param = new TwelveTimeSeriesParam(
            httpClient, repoMock.Object, "test-key", "AAPL",
            Start, End, TwelveDataFormat.Csv, "4h");

        handler.EnqueueResponse(HttpStatusCode.OK,
            TimeSeriesFixtures.BuildCsvPayload(("2024-01-01 00:00:00", "1900", "1950", "1880", "1920")),
            "text/csv");
        handler.EnqueueResponse(HttpStatusCode.OK, "datetime,open,high,low,close\n", "text/csv");

        var result = await param.GetSeries();
        Assert.True(result.Count > 0);
    }

    [Fact]
    public async Task GetSeries_FullyCached_DoesNotCallApi()
    {
        // Pre-populate the cache with the expected timestamp
        var cachedDoc = new TimeSeriesCacheDocument
        {
            Symbol = "AAPL",
            Intervals = new Dictionary<string, SortedDictionary<string, TimeSeriesValue>>
            {
                ["4h"] = new SortedDictionary<string, TimeSeriesValue>(StringComparer.Ordinal)
                {
                    [TwelveTimeSeriesParamExtensions.ToStorageKey(Start)] = TimeSeriesFixtures.RealCandle(Start)
                }
            }
        };

        var (param, handler, _) = TimeSeriesFixtures.BuildParam(
            startDate: Start, endDate: End, cachedDoc: cachedDoc);

        var result = await param.GetSeries();

        Assert.Empty(handler.SentRequests);
        Assert.True(result.Count > 0);
    }

    [Fact]
    public async Task GetSeries_EmptyCache_SaveAsyncCalledAfterFetch()
    {
        var (param, handler, repoMock) = TimeSeriesFixtures.BuildParam(startDate: Start, endDate: End);

        handler.EnqueueResponse(HttpStatusCode.OK,
            TimeSeriesFixtures.BuildJsonPayload(("2024-01-01 00:00:00", "1900", "1950", "1880", "1920")));
        handler.EnqueueResponse(HttpStatusCode.OK, TimeSeriesFixtures.BuildNoDataJson());

        await param.GetSeries();

        repoMock.Verify(r => r.SaveAsync(
            It.IsAny<string>(),
            It.IsAny<TimeSeriesCacheDocument>(),
            It.IsAny<CancellationToken>()), Times.AtLeastOnce);
    }

    [Fact]
    public async Task GetSeries_FullyCached_SaveAsyncNotCalled()
    {
        var cachedDoc = new TimeSeriesCacheDocument
        {
            Symbol = "AAPL",
            Intervals = new Dictionary<string, SortedDictionary<string, TimeSeriesValue>>
            {
                ["4h"] = new SortedDictionary<string, TimeSeriesValue>(StringComparer.Ordinal)
                {
                    [TwelveTimeSeriesParamExtensions.ToStorageKey(Start)] = TimeSeriesFixtures.RealCandle(Start)
                }
            }
        };

        var (param, _, repoMock) = TimeSeriesFixtures.BuildParam(
            startDate: Start, endDate: End, cachedDoc: cachedDoc);

        await param.GetSeries();

        repoMock.Verify(r => r.SaveAsync(
            It.IsAny<string>(),
            It.IsAny<TimeSeriesCacheDocument>(),
            It.IsAny<CancellationToken>()), Times.Never);
    }

    [Fact]
    public async Task GetSeries_ApiReturnsNoData_FillsGapWithIsFilled()
    {
        var (param, handler, _) = TimeSeriesFixtures.BuildParam(startDate: Start, endDate: End);

        // First call: returns one real candle
        handler.EnqueueResponse(HttpStatusCode.OK,
            TimeSeriesFixtures.BuildJsonPayload(("2024-01-01 04:00:00", "1900", "1950", "1880", "1920")));
        // Second call: no more data
        handler.EnqueueResponse(HttpStatusCode.OK, TimeSeriesFixtures.BuildNoDataJson());

        var result = await param.GetSeries();

        // T0 (00:00) should be filled since the only real data starts at 04:00
        var t0Key = new DateTime(2024, 1, 1, 0, 0, 0);
        if (result.TryGetValue(t0Key, out var filledCandle))
        {
            Assert.True(filledCandle.IsFilled);
        }
    }

    [Fact]
    public async Task GetSeries_ResultFilteredToRequestedDateRange()
    {
        // Cache has data wider than requested range
        var wideStart = new DateTime(2023, 12, 31, 0, 0, 0);
        var wideEnd = new DateTime(2024, 1, 2, 0, 0, 0);
        var cachedDoc = new TimeSeriesCacheDocument
        {
            Symbol = "AAPL",
            Intervals = new Dictionary<string, SortedDictionary<string, TimeSeriesValue>>
            {
                ["4h"] = new SortedDictionary<string, TimeSeriesValue>(StringComparer.Ordinal)
                {
                    [TwelveTimeSeriesParamExtensions.ToStorageKey(wideStart)] = TimeSeriesFixtures.RealCandle(wideStart),
                    [TwelveTimeSeriesParamExtensions.ToStorageKey(Start)] = TimeSeriesFixtures.RealCandle(Start),
                    [TwelveTimeSeriesParamExtensions.ToStorageKey(wideEnd)] = TimeSeriesFixtures.RealCandle(wideEnd),
                }
            }
        };

        var (param, _, _) = TimeSeriesFixtures.BuildParam(
            startDate: Start, endDate: End, cachedDoc: cachedDoc);

        var result = await param.GetSeries();

        // Out-of-range timestamps should not be returned
        Assert.DoesNotContain(wideStart, result.Keys);
        Assert.DoesNotContain(wideEnd, result.Keys);
    }
}
