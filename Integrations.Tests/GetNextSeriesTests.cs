using System.Net;
using Integrations.TwelveData;
using Integrations.Tests.Helpers;

namespace Integrations.Tests;

public class GetNextSeriesTests
{
    private static readonly DateTime Start = new DateTime(2024, 1, 1, 0, 0, 0);
    private static readonly DateTime End = new DateTime(2024, 3, 1, 0, 0, 0);

    [Fact]
    public async Task GetNextSeries_NullParam_ThrowsArgumentNullException()
    {
        TwelveTimeSeriesParam? param = null;
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            param!.GetNextSeries(new Dictionary<DateTime, TimeSeriesValue>()));
    }

    [Fact]
    public async Task GetNextSeries_NullCurrentBatch_ThrowsArgumentNullException()
    {
        var (param, _, _) = TimeSeriesFixtures.BuildParam(startDate: Start, endDate: End);
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            param.GetNextSeries(null!));
    }

    [Fact]
    public async Task GetNextSeries_EmptyCurrentBatch_ReturnsEmpty()
    {
        var (param, handler, _) = TimeSeriesFixtures.BuildParam(startDate: Start, endDate: End);

        var result = await param.GetNextSeries(new Dictionary<DateTime, TimeSeriesValue>());

        Assert.Empty(result);
        Assert.Empty(handler.SentRequests);
    }

    [Fact]
    public async Task GetNextSeries_OldestKeyAtStartDate_ReturnsEmpty()
    {
        var (param, handler, _) = TimeSeriesFixtures.BuildParam(startDate: Start, endDate: End);

        var batch = new Dictionary<DateTime, TimeSeriesValue>
        {
            [Start] = TimeSeriesFixtures.RealCandle(Start) // oldest == startDate
        };

        var result = await param.GetNextSeries(batch);

        Assert.Empty(result);
        Assert.Empty(handler.SentRequests);
    }

    [Fact]
    public async Task GetNextSeries_OldestKeyBeforeStartDate_ReturnsEmpty()
    {
        var (param, handler, _) = TimeSeriesFixtures.BuildParam(startDate: Start, endDate: End);

        var batch = new Dictionary<DateTime, TimeSeriesValue>
        {
            [Start.AddDays(-1)] = TimeSeriesFixtures.RealCandle(Start.AddDays(-1)) // oldest < startDate
        };

        var result = await param.GetNextSeries(batch);

        Assert.Empty(result);
        Assert.Empty(handler.SentRequests);
    }

    [Fact]
    public async Task GetNextSeries_OldestKeyAfterStartDate_CallsGetSeries()
    {
        var (param, handler, _) = TimeSeriesFixtures.BuildParam(startDate: Start, endDate: End);

        // oldest = Feb 1, which is after Start (Jan 1) → should fetch Jan 1 to Jan 31 23:59...
        var oldest = new DateTime(2024, 2, 1, 0, 0, 0);
        var batch = new Dictionary<DateTime, TimeSeriesValue>
        {
            [oldest] = TimeSeriesFixtures.RealCandle(oldest),
            [oldest.AddHours(4)] = TimeSeriesFixtures.RealCandle(oldest.AddHours(4))
        };

        // Enqueue enough responses to satisfy the fetching loop (no-data terminates)
        handler.EnqueueResponse(System.Net.HttpStatusCode.OK, TimeSeriesFixtures.BuildNoDataJson());

        var result = await param.GetNextSeries(batch);

        // At least one HTTP request should have been made
        Assert.True(handler.SentRequests.Count > 0);
    }
}
