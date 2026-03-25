using Importer;
using Integrations.TwelveData;
using Xunit;

namespace Importer.Tests;

/// <summary>
/// Unit tests for <see cref="Transformer.BuildFeatureMatrix"/>.
/// Strategy: supply only XAU/USD with a handful of 4h candles and no
/// cross-asset data, so the column names and feature values are predictable.
/// </summary>
public class BuildFeatureMatrixTests
{
    // Three 4h candles: T0, T1, T2 with steadily rising closes 100 / 101 / 102
    private static readonly DateTime T0 = new(2024, 1, 1,  0, 0, 0);
    private static readonly DateTime T1 = new(2024, 1, 1,  4, 0, 0);
    private static readonly DateTime T2 = new(2024, 1, 1,  8, 0, 0);

    /// <summary>
    /// Minimal config: single period per indicator family to keep column count
    /// deterministic and avoid warm-up issues with a 3-candle series.
    /// Only 4h target horizon so there is exactly one target column.
    /// </summary>
    private static DatasetConfig MinimalCfg(DateTime? trainingStart = null) => new()
    {
        TargetSymbol       = "XAU/USD",
        TargetHorizons     = [("4h", 1)],
        AggStats           = ["mean"],
        SmaPeriods         = [2],
        EmaPeriods         = [2],
        RsiPeriods         = [2],
        RocPeriods         = [1],
        StdDevPeriods      = [2],
        AtrPeriods         = [2],
        BbPeriods          = [2],
        CciPeriods         = [2],
        WilliamsRPeriods   = [2],
        StochKPeriods      = [2],
        StochDPeriod       = 1,
        MacdFast           = 2,
        MacdSlow           = 3,
        MacdSignal         = 2,
        DistN              = 2,
        AdZScorePeriod     = 2,
        TrainingStartDate  = trainingStart,
    };

    private static TimeSeriesValue Candle(DateTime dt, decimal close) =>
        new(dt, close, close, close, close, IsFilled: false);

    /// <summary>
    /// 3 candles for XAU/USD at 4h with closes 100, 101, 102.
    /// allCandles["XAU/USD"]["4h"] = [T0, T1, T2]
    /// </summary>
    private static Dictionary<string, Dictionary<string, List<TimeSeriesValue>>> ThreeCandles() =>
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["XAU/USD"] = new Dictionary<string, List<TimeSeriesValue>>(StringComparer.Ordinal)
            {
                ["4h"] = [Candle(T0, 100m), Candle(T1, 101m), Candle(T2, 102m)]
            }
        };

    private static Dictionary<string, Dictionary<string, List<IndicatorValue>>> EmptyIndicators() =>
        new(StringComparer.OrdinalIgnoreCase);

    // ── Row count ─────────────────────────────────────────────────────────────

    [Fact]
    public void BuildFeatureMatrix_ThreeCandles_ReturnsThreeRows()
    {
        // All 3 bars are emitted. T2 has no future bar so its target return is null,
        // but the row itself is still included (hasTarget is always true for base-TF bars).
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        Assert.Equal(3, rows.Count);
    }

    [Fact]
    public void BuildFeatureMatrix_NoFourHourData_ReturnsEmptyRows()
    {
        var allCandles = new Dictionary<string, Dictionary<string, List<TimeSeriesValue>>>(
            StringComparer.OrdinalIgnoreCase)
        {
            ["XAU/USD"] = new Dictionary<string, List<TimeSeriesValue>>(StringComparer.Ordinal)
            {
                // Only "1day" — no "4h" key, so baseSd is null
                ["1day"] = [Candle(T0, 100m)]
            }
        };

        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), allCandles, EmptyIndicators());

        Assert.Empty(rows);
    }

    [Fact]
    public void BuildFeatureMatrix_EmptyAllCandles_ReturnsEmptyRows()
    {
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(),
            new Dictionary<string, Dictionary<string, List<TimeSeriesValue>>>(StringComparer.OrdinalIgnoreCase),
            EmptyIndicators());

        Assert.Empty(rows);
    }

    // ── Row timestamps ────────────────────────────────────────────────────────

    [Fact]
    public void BuildFeatureMatrix_RowTimestampsMatchCandleTimes()
    {
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        Assert.Equal(T0, rows[0].Timestamp);
        Assert.Equal(T1, rows[1].Timestamp);
    }

    // ── Column names ──────────────────────────────────────────────────────────

    [Fact]
    public void BuildFeatureMatrix_ColumnsContainTimestamp()
    {
        var (_, columns, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        Assert.Contains("Timestamp", columns);
    }

    [Fact]
    public void BuildFeatureMatrix_ColumnsContainTargetReturn()
    {
        var (_, columns, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        Assert.Contains("XAUUSD_Target_4h_Return", columns);
    }

    [Fact]
    public void BuildFeatureMatrix_ColumnsContainXauUsd4hLogReturn()
    {
        var (_, columns, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        Assert.Contains("XAUUSD_4h_LogReturn", columns);
    }

    [Fact]
    public void BuildFeatureMatrix_ColumnsContainPeriodSuffixedIndicators()
    {
        var (_, columns, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        Assert.Contains("XAUUSD_4h_SMA_2", columns);
        Assert.Contains("XAUUSD_4h_EMA_2", columns);
        Assert.Contains("XAUUSD_4h_RSI_2", columns);
        Assert.Contains("XAUUSD_4h_ATR_2", columns);
        Assert.Contains("XAUUSD_4h_BbUpper_2", columns);
        Assert.Contains("XAUUSD_4h_BbLower_2", columns);
        Assert.Contains("XAUUSD_4h_BbWidth_2", columns);
        Assert.Contains("XAUUSD_4h_StochK_2", columns);
        Assert.Contains("XAUUSD_4h_StochD_2", columns);
    }

    [Fact]
    public void BuildFeatureMatrix_ColumnsContainTimeCols()
    {
        var (_, columns, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        Assert.Contains("HourOfDay_Sin", columns);
        Assert.Contains("HourOfDay_Cos", columns);
        Assert.Contains("DayOfWeek_Sin", columns);
        Assert.Contains("DayOfWeek_Cos", columns);
        Assert.Contains("Session", columns);
    }

    [Fact]
    public void BuildFeatureMatrix_ColumnsContainCrossAssetCols()
    {
        var (_, columns, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        Assert.Contains("GoldSilverRatio", columns);
        Assert.Contains("GoldOilRatio", columns);
        Assert.Contains("YieldCurveProxy", columns);
    }

    // ── Values in rows ────────────────────────────────────────────────────────

    [Fact]
    public void BuildFeatureMatrix_RowValues_ContainAllColumns()
    {
        var (rows, columns, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        foreach (var row in rows)
            foreach (var col in columns)
                Assert.True(row.Values.ContainsKey(col),
                    $"Row {row.Timestamp:yyyy-MM-dd HH:mm:ss} is missing column '{col}'");
    }

    [Fact]
    public void BuildFeatureMatrix_TargetReturn_Row0_EqualsLogReturn_T1_over_T0()
    {
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        // Row 0 targets T1 close / T0 close = log(101/100)
        var expected = Math.Log(101.0 / 100.0);
        var actual   = rows[0].Values["XAUUSD_Target_4h_Return"];

        Assert.NotNull(actual);
        Assert.Equal(expected, actual!.Value, precision: 10);
    }

    [Fact]
    public void BuildFeatureMatrix_TargetReturn_Row1_EqualsLogReturn_T2_over_T1()
    {
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        var expected = Math.Log(102.0 / 101.0);
        var actual   = rows[1].Values["XAUUSD_Target_4h_Return"];

        Assert.NotNull(actual);
        Assert.Equal(expected, actual!.Value, precision: 10);
    }

    [Fact]
    public void BuildFeatureMatrix_TimeEncoding_Row0_CorrectForMidnight()
    {
        // T0 = 2024-01-01 00:00 UTC
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        // hour = 0 → sin(0) = 0, cos(0) = 1
        Assert.Equal(0.0, rows[0].Values["HourOfDay_Sin"]!.Value, precision: 10);
        Assert.Equal(1.0, rows[0].Values["HourOfDay_Cos"]!.Value, precision: 10);
    }

    [Fact]
    public void BuildFeatureMatrix_TimeEncoding_Row1_CorrectFor4AM()
    {
        // T1 = 2024-01-01 04:00 UTC → hour = 4
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        var expectedSin = Math.Sin(2 * Math.PI * 4.0 / 24.0);
        var expectedCos = Math.Cos(2 * Math.PI * 4.0 / 24.0);
        Assert.Equal(expectedSin, rows[1].Values["HourOfDay_Sin"]!.Value, precision: 10);
        Assert.Equal(expectedCos, rows[1].Values["HourOfDay_Cos"]!.Value, precision: 10);
    }

    // ── TrainingStartDate filtering ───────────────────────────────────────────

    [Fact]
    public void BuildFeatureMatrix_TrainingStartDate_FiltersEarlierRows()
    {
        // Start at T1 → T0 is excluded; T1 and T2 are both emitted
        // (T2 has a null target return but is still a valid row)
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(trainingStart: T1), ThreeCandles(), EmptyIndicators());

        Assert.Equal(2, rows.Count);
        Assert.Equal(T1, rows[0].Timestamp);
        Assert.Equal(T2, rows[1].Timestamp);
    }

    [Fact]
    public void BuildFeatureMatrix_TrainingStartDate_ExactBoundaryIsIncluded()
    {
        // T0 is exactly the TrainingStartDate → all 3 rows are included
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(trainingStart: T0), ThreeCandles(), EmptyIndicators());

        Assert.Equal(3, rows.Count);
        Assert.Equal(T0, rows[0].Timestamp);
    }

    [Fact]
    public void BuildFeatureMatrix_TrainingStartDate_AfterAllBars_ReturnsEmpty()
    {
        var future = T2.AddHours(4);
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            MinimalCfg(trainingStart: future), ThreeCandles(), EmptyIndicators());

        Assert.Empty(rows);
    }

    // ── ColumnStats ───────────────────────────────────────────────────────────

    [Fact]
    public void BuildFeatureMatrix_ColumnStats_KeysMatchColumns()
    {
        var (_, columns, columnStats) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        foreach (var col in columns)
            Assert.True(columnStats.ContainsKey(col),
                $"ColumnStats is missing key '{col}'");
    }

    [Fact]
    public void BuildFeatureMatrix_ColumnStats_EmptyCountGteNullCount()
    {
        var (_, columns, columnStats) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        foreach (var col in columns)
        {
            var (nullCount, emptyCount, zeroCount) = columnStats[col];
            Assert.True(emptyCount >= nullCount,
                $"Column '{col}': EmptyCount ({emptyCount}) < NullCount ({nullCount})");
            Assert.True(emptyCount == nullCount + zeroCount,
                $"Column '{col}': EmptyCount ({emptyCount}) != NullCount + ZeroCount ({nullCount + zeroCount})");
        }
    }

    [Fact]
    public void BuildFeatureMatrix_ColumnStats_TargetReturn_OneNullForLastBar()
    {
        // T0→T1 and T1→T2 have valid targets; T2 has no future bar → null target
        var (_, _, columnStats) = Transformer.BuildFeatureMatrix(
            MinimalCfg(), ThreeCandles(), EmptyIndicators());

        var (nullCount, _, _) = columnStats["XAUUSD_Target_4h_Return"];
        Assert.Equal(1, nullCount);
    }

    // ── InvalidArgument guard ─────────────────────────────────────────────────

    [Fact]
    public void BuildFeatureMatrix_InvalidTargetSymbol_ThrowsArgumentException()
    {
        var cfg = new DatasetConfig { TargetSymbol = "INVALID_SYMBOL_XYZ" };

        Assert.Throws<ArgumentException>(() =>
            Transformer.BuildFeatureMatrix(cfg, ThreeCandles(), EmptyIndicators()));
    }
}
