using System.Text;
using Integrations.TwelveData;
using Moq;

namespace Integrations.Tests.Helpers;

public static class TimeSeriesFixtures
{
    public static TimeSeriesValue RealCandle(
        DateTime dt,
        decimal open = 1900m,
        decimal high = 1950m,
        decimal low = 1880m,
        decimal close = 1920m)
        => new(dt, open, high, low, close, IsFilled: false);

    public static TimeSeriesValue FilledCandle(
        DateTime dt,
        decimal open = 1900m,
        decimal high = 1950m,
        decimal low = 1880m,
        decimal close = 1920m)
        => new(dt, open, high, low, close, IsFilled: true);

    public static string BuildJsonPayload(
        params (string datetime, string open, string high, string low, string close)[] rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("{\"values\":[");

        for (int i = 0; i < rows.Length; i++)
        {
            var (dt, o, h, l, c) = rows[i];
            sb.Append($"{{\"datetime\":\"{dt}\",\"open\":\"{o}\",\"high\":\"{h}\",\"low\":\"{l}\",\"close\":\"{c}\"}}");
            if (i < rows.Length - 1)
                sb.AppendLine(",");
        }

        sb.AppendLine("]}");
        return sb.ToString();
    }

    public static string BuildCsvPayload(
        params (string datetime, string open, string high, string low, string close)[] rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("datetime,open,high,low,close");
        foreach (var (dt, o, h, l, c) in rows)
            sb.AppendLine($"{dt},{o},{h},{l},{c}");
        return sb.ToString();
    }

    public static string BuildNoDataJson()
        => """{"status":"error","code":400,"message":"No data is available on the specified dates."}""";

    public static (
        TwelveTimeSeriesParam param,
        FakeHttpMessageHandler handler,
        Mock<IRepository<TimeSeriesCacheDocument>> repoMock)
        BuildParam(
            string symbol = "AAPL",
            string interval = "4h",
            DateTime? startDate = null,
            DateTime? endDate = null,
            TimeSeriesCacheDocument? cachedDoc = null)
    {
        var handler = new FakeHttpMessageHandler();
        var httpClient = new HttpClient(handler)
        {
            BaseAddress = new Uri("https://api.twelvedata.com/")
        };

        var repoMock = new Mock<IRepository<TimeSeriesCacheDocument>>();
        repoMock
            .Setup(r => r.GetAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(cachedDoc);
        repoMock
            .Setup(r => r.SaveAsync(It.IsAny<string>(), It.IsAny<TimeSeriesCacheDocument>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        var start = startDate ?? new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var end = endDate ?? new DateTime(2024, 1, 2, 0, 0, 0, DateTimeKind.Utc);

        var param = new TwelveTimeSeriesParam(
            httpClient: httpClient,
            repository: repoMock.Object,
            apiKey: "test-key",
            symbol: symbol,
            startDate: start,
            endDate: end,
            format: TwelveDataFormat.Json,
            interval: interval);

        return (param, handler, repoMock);
    }
}
