using Importer;
using Integrations.TwelveData;
using Xunit;

namespace Importer.Tests;

/// <summary>
/// Tests for the vol scalar behaviour introduced to normalise the target return.
///
/// Key invariants:
///   1. When realized vol is zero (flat prices), the target and TargetVolScalar
///      columns are null — the row is excluded from training rather than having
///      its target inflated to millions (the pre-fix behaviour with a 1e-8 floor).
///   2. When realized vol is below the reliability threshold (≤ 1e-5), same as (1).
///   3. When realized vol is normal (> 1e-5), the target equals rawReturn / rv
///      with no floor applied.
///   4. The TargetVolScalar column echoes the vol value used for (3) and is null
///      for cases (1) and (2).
/// </summary>
public class VolScalarTests
{
    private static readonly DateTime Base = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    private const int Period = 10;   // vol period used in all fixtures
    private const int NBars  = 25;   // enough bars: period warm-up + target bar + headroom

    /// <summary>
    /// Minimal config: period-10 vol scalar, no indicators, no cross-asset ratios.
    /// FeatureLag = 0 (default), so cutoffBi == bi (vol at row i uses closes 0..i).
    /// </summary>
    private static DatasetConfig VolCfg() => new()
    {
        TargetSymbol      = "XAU/USD",
        TargetHorizons    = [("4h", 1)],
        ShortTimeframes   = [],
        LongTimeframes    = ["4h"],
        AggStats          = [],
        EmaPeriods        = [],
        SmaPeriods        = [],
        RsiPeriods        = [],
        RocPeriods        = [],
        StdDevPeriods     = [],
        AtrPeriods        = [],
        BbPeriods         = [],
        BbWidthZPeriod    = 0,
        CciPeriods        = [],
        WilliamsRPeriods  = [],
        StochKPeriods     = [],
        StochDPeriod      = 0,
        MacdFast          = 0,
        MacdSlow          = 0,
        MacdSignal        = 0,
        DistN             = 0,
        AdZScorePeriod    = 0,
        RealizedVolConfig = new() { ["XAU/USD"] = new() { ["4h"] = [Period] } },
        VolRatioMaPeriods = [0],   // vol ratio disabled
        Ratios            = [],    // no cross-asset ratios needed
    };

    private static Dictionary<string, Dictionary<string, List<TimeSeriesValue>>> MakeCandles(
        decimal[] closes) =>
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["XAU/USD"] = new(StringComparer.Ordinal)
            {
                ["4h"] = closes.Select((c, i) => new TimeSeriesValue(
                    Base.AddHours(i * 4), c, c * 1.01m, c * 0.99m, c, IsFilled: false))
                    .ToList()
            }
        };

    private static Dictionary<string, Dictionary<string, List<IndicatorValue>>> EmptyIndicators() =>
        new(StringComparer.OrdinalIgnoreCase);

    // ── Zero vol (flat prices) ────────────────────────────────────────────────

    /// <summary>
    /// When all closes are identical the realized vol is exactly zero.
    /// The target column must be null: the row should be excluded from training
    /// rather than having its target inflated to rawReturn / 1e-8.
    /// </summary>
    [Fact]
    public void ZeroVol_FlatPrices_TargetIsNull()
    {
        var closes = Enumerable.Repeat(2000m, NBars).ToArray();
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            VolCfg(), MakeCandles(closes), EmptyIndicators());

        // First bar with a computable vol is bar (Period-1). Check that bar and
        // a few subsequent ones all have null targets.
        foreach (var bar in new[] { Period - 1, Period, Period + 3 })
        {
            var row = rows.First(r => r.Timestamp == Base.AddHours(bar * 4));
            Assert.Null(row.Values["XAUUSD_Target_4h_Return"]);
        }
    }

    /// <summary>
    /// TargetVolScalar column must also be null when vol is zero — it must not
    /// expose the 1e-8 sentinel that would indicate a floor was applied.
    /// </summary>
    [Fact]
    public void ZeroVol_FlatPrices_TargetVolScalarIsNull()
    {
        var closes = Enumerable.Repeat(2000m, NBars).ToArray();
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            VolCfg(), MakeCandles(closes), EmptyIndicators());

        var row = rows.First(r => r.Timestamp == Base.AddHours((Period - 1) * 4));
        Assert.Null(row.Values["XAUUSD_TargetVolScalar"]);
    }

    // ── Near-zero vol (below reliability threshold) ───────────────────────────

    /// <summary>
    /// Prices that change by ~1e-9 per bar produce a vol far below the 1e-5
    /// threshold. Target must be null, not a huge inflated number.
    /// </summary>
    [Fact]
    public void NearZeroVol_BelowThreshold_TargetIsNull()
    {
        // Epsilon per bar ≈ 1e-9 in log-return space → std dev ≪ 1e-5
        var closes = Enumerable.Range(0, NBars)
            .Select(i => 2000m + i * 0.000_000_001m)
            .ToArray();

        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            VolCfg(), MakeCandles(closes), EmptyIndicators());

        var row = rows.First(r => r.Timestamp == Base.AddHours((Period - 1) * 4));
        Assert.Null(row.Values["XAUUSD_Target_4h_Return"]);
    }

    // ── Normal vol ────────────────────────────────────────────────────────────

    /// <summary>
    /// With realistic price variation, the target equals rawReturn / realizedVol
    /// exactly (no 1e-8 floor, no other transformation).
    /// </summary>
    [Fact]
    public void NormalVol_Target_EqualsRawReturnDividedByVol()
    {
        // Realistic gold-like prices: drift + oscillation → vol ≈ 0.003
        var closes = Enumerable.Range(0, NBars)
            .Select(i => 2000m + i * 0.5m + (decimal)Math.Sin(i * 0.7) * 3m)
            .ToArray();

        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            VolCfg(), MakeCandles(closes), EmptyIndicators());

        // Bar (Period-1): first bar whose rv window is fully populated (bars 0..Period-1)
        int testBar = Period - 1;
        var row = rows.First(r => r.Timestamp == Base.AddHours(testBar * 4));

        // Manually reproduce ComputeRv over bars 0..testBar (log returns 1..testBar)
        var logRets = Enumerable.Range(1, testBar)
            .Select(i => Math.Log((double)closes[i] / (double)closes[i - 1]))
            .ToArray();
        double mean        = logRets.Average();
        double variance    = logRets.Average(v => (v - mean) * (v - mean));
        double rv          = Math.Sqrt(variance);

        double rawReturn   = Math.Log((double)closes[testBar + 1] / (double)closes[testBar]);
        double expected    = rawReturn / rv;

        Assert.NotNull(row.Values["XAUUSD_Target_4h_Return"]);
        Assert.True(rv > 1e-5, $"Fixture vol {rv} is not above threshold — test is misconfigured");
        Assert.Equal(expected, row.Values["XAUUSD_Target_4h_Return"]!.Value, precision: 10);
    }

    /// <summary>
    /// TargetVolScalar column must match the realized vol that was applied —
    /// i.e. the population std dev of log returns, with no floor transformation.
    /// </summary>
    [Fact]
    public void NormalVol_TargetVolScalar_MatchesComputedVol()
    {
        var closes = Enumerable.Range(0, NBars)
            .Select(i => 2000m + i * 0.5m + (decimal)Math.Sin(i * 0.7) * 3m)
            .ToArray();

        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            VolCfg(), MakeCandles(closes), EmptyIndicators());

        int testBar = Period - 1;
        var row = rows.First(r => r.Timestamp == Base.AddHours(testBar * 4));

        var logRets = Enumerable.Range(1, testBar)
            .Select(i => Math.Log((double)closes[i] / (double)closes[i - 1]))
            .ToArray();
        double mean        = logRets.Average();
        double variance    = logRets.Average(v => (v - mean) * (v - mean));
        double expectedVol = Math.Sqrt(variance);

        Assert.NotNull(row.Values["XAUUSD_TargetVolScalar"]);
        Assert.Equal(expectedVol, row.Values["XAUUSD_TargetVolScalar"]!.Value, precision: 10);
    }

    // ── Warm-up period ────────────────────────────────────────────────────────

    /// <summary>
    /// Bars before the vol warm-up (indices 0 .. Period-2) cannot have a vol
    /// scalar because the rolling window is not yet full. Their targets must be
    /// null regardless of price movements.
    /// </summary>
    [Fact]
    public void Rows_BeforeVolWarmup_HaveNullTarget()
    {
        var closes = Enumerable.Range(0, NBars)
            .Select(i => 2000m + i * 0.5m + (decimal)Math.Sin(i * 0.7) * 3m)
            .ToArray();

        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            VolCfg(), MakeCandles(closes), EmptyIndicators());

        for (int bar = 0; bar < Period - 1; bar++)
        {
            var ts  = Base.AddHours(bar * 4);
            var row = rows.FirstOrDefault(r => r.Timestamp == ts);
            if (row is null) continue;

            Assert.Null(row.Values["XAUUSD_Target_4h_Return"]);
        }
    }

    /// <summary>
    /// The bar at index (Period-1) is the first row that can have a non-null
    /// target — confirms the warm-up boundary is off by zero, not off by one.
    /// </summary>
    [Fact]
    public void FirstBarAfterWarmup_HasNonNullTarget()
    {
        var closes = Enumerable.Range(0, NBars)
            .Select(i => 2000m + i * 0.5m + (decimal)Math.Sin(i * 0.7) * 3m)
            .ToArray();

        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            VolCfg(), MakeCandles(closes), EmptyIndicators());

        var ts  = Base.AddHours((Period - 1) * 4);
        var row = rows.First(r => r.Timestamp == ts);

        Assert.NotNull(row.Values["XAUUSD_Target_4h_Return"]);
    }
}
