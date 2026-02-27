using Integrations.TwelveData;

namespace Integrations.Tests;

public class ParseIntervalTests
{
    [Fact] public void ParseInterval_1min_Returns1Minute()   => Assert.Equal(TimeSpan.FromMinutes(1),  TwelveTimeSeriesParamExtensions.ParseInterval("1min"));
    [Fact] public void ParseInterval_5min_Returns5Minutes()  => Assert.Equal(TimeSpan.FromMinutes(5),  TwelveTimeSeriesParamExtensions.ParseInterval("5min"));
    [Fact] public void ParseInterval_15min_Returns15Minutes()=> Assert.Equal(TimeSpan.FromMinutes(15), TwelveTimeSeriesParamExtensions.ParseInterval("15min"));
    [Fact] public void ParseInterval_30min_Returns30Minutes()=> Assert.Equal(TimeSpan.FromMinutes(30), TwelveTimeSeriesParamExtensions.ParseInterval("30min"));
    [Fact] public void ParseInterval_45min_Returns45Minutes()=> Assert.Equal(TimeSpan.FromMinutes(45), TwelveTimeSeriesParamExtensions.ParseInterval("45min"));
    [Fact] public void ParseInterval_1h_Returns1Hour()       => Assert.Equal(TimeSpan.FromHours(1),    TwelveTimeSeriesParamExtensions.ParseInterval("1h"));
    [Fact] public void ParseInterval_2h_Returns2Hours()      => Assert.Equal(TimeSpan.FromHours(2),    TwelveTimeSeriesParamExtensions.ParseInterval("2h"));
    [Fact] public void ParseInterval_4h_Returns4Hours()      => Assert.Equal(TimeSpan.FromHours(4),    TwelveTimeSeriesParamExtensions.ParseInterval("4h"));
    [Fact] public void ParseInterval_5h_Returns5Hours()      => Assert.Equal(TimeSpan.FromHours(5),    TwelveTimeSeriesParamExtensions.ParseInterval("5h"));
    [Fact] public void ParseInterval_1day_Returns1Day()      => Assert.Equal(TimeSpan.FromDays(1),     TwelveTimeSeriesParamExtensions.ParseInterval("1day"));
    [Fact] public void ParseInterval_1week_Returns7Days()    => Assert.Equal(TimeSpan.FromDays(7),     TwelveTimeSeriesParamExtensions.ParseInterval("1week"));

    [Fact]
    public void ParseInterval_1month_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() => TwelveTimeSeriesParamExtensions.ParseInterval("1month"));
    }

    [Fact]
    public void ParseInterval_UnknownInterval_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() => TwelveTimeSeriesParamExtensions.ParseInterval("3h"));
    }

    [Fact]
    public void ParseInterval_UppercaseInterval_ThrowsInvalidOperationException()
    {
        // String matching is case-sensitive; "4H" is not "4h"
        Assert.Throws<InvalidOperationException>(() => TwelveTimeSeriesParamExtensions.ParseInterval("4H"));
    }

    [Fact]
    public void ParseInterval_EmptyString_ThrowsInvalidOperationException()
    {
        Assert.Throws<InvalidOperationException>(() => TwelveTimeSeriesParamExtensions.ParseInterval(""));
    }
}
