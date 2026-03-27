using Importer;
using Integrations.TwelveData;
using Xunit;

namespace Importer.Tests;

/// <summary>
/// Verifies that no feature value for a row at timestamp T uses data from
/// timestamps strictly after the target bar (T + 1 bar). Tests work by
/// rebuilding the dataset with a modified future candle and asserting that
/// all feature columns are identical while the target changes as expected.
/// </summary>
public class DataLeakageTests
{
    // 30 candles: bars 0–29, each 4h apart, starting 2020-01-01 00:00 UTC
    private static readonly DateTime Base = new(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    /// <summary>
    /// Minimal config for leakage tests:
    ///   - RealizedVolPeriod = 20 (matches production; first vol-normalised row at bar 19)
    ///   - EMA(5) as a representative backward-looking indicator
    ///   - All other indicators disabled to keep column count small
    /// </summary>
    private static DatasetConfig LeakageCfg() => new()
    {
        TargetSymbol      = "XAU/USD",
        TargetHorizons    = [("4h", 1)],
        AggStats          = ["mean"],
        EmaPeriods        = [5],
        SmaPeriods        = [],
        RsiPeriods        = [],
        RocPeriods        = [],
        StdDevPeriods     = [],
        AtrPeriods        = [],
        BbPeriods         = [],
        CciPeriods        = [],
        WilliamsRPeriods  = [],
        StochKPeriods     = [],
        StochDPeriod      = 0,
        MacdFast          = 0,
        MacdSlow          = 0,
        MacdSignal        = 0,
        DistN             = 0,
        AdZScorePeriod    = 0,
        RealizedVolPeriod = 20,
        VolRatioMaPeriod  = 0,
        WalkForwardFolds  = 1,
    };

    /// <summary>Builds an allCandles dict from a flat array of closes.</summary>
    private static Dictionary<string, Dictionary<string, List<TimeSeriesValue>>> MakeCandles(
        decimal[] closes)
    {
        var candles = closes
            .Select((c, i) => new TimeSeriesValue(
                Base.AddHours(i * 4), c, c * 1.01m, c * 0.99m, c, IsFilled: false))
            .ToList();

        return new(StringComparer.OrdinalIgnoreCase)
        {
            ["XAU/USD"] = new(StringComparer.Ordinal) { ["4h"] = candles }
        };
    }

    /// <summary>
    /// Builds an allCandles dict with XAU/USD, XAG/USD, and WTI/USD — all at
    /// the same 4h timestamps — so cross-asset ratio columns are populated.
    /// </summary>
    private static Dictionary<string, Dictionary<string, List<TimeSeriesValue>>> MakeCrossAssetCandles(
        decimal[] xauCloses, decimal[] xagCloses, decimal[] wtiCloses)
    {
        List<TimeSeriesValue> Make(decimal[] closes) =>
            closes.Select((c, i) => new TimeSeriesValue(
                Base.AddHours(i * 4), c, c * 1.01m, c * 0.99m, c, IsFilled: false))
            .ToList();

        return new(StringComparer.OrdinalIgnoreCase)
        {
            ["XAU/USD"] = new(StringComparer.Ordinal) { ["4h"] = Make(xauCloses) },
            ["XAG/USD"] = new(StringComparer.Ordinal) { ["4h"] = Make(xagCloses) },
            ["WTI/USD"] = new(StringComparer.Ordinal) { ["4h"] = Make(wtiCloses) },
        };
    }

    private static Dictionary<string, Dictionary<string, List<IndicatorValue>>> EmptyIndicators() =>
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Generates n deterministic closes that vary enough for indicators to be
    /// non-trivial: a slow drift plus a small oscillation.
    /// </summary>
    private static decimal[] GenerateCloses(int n) =>
        Enumerable.Range(0, n)
            .Select(i => 2000m + i * 0.5m + (decimal)Math.Sin(i * 0.7) * 3m)
            .ToArray();

    // ── Helper ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Compares every feature column (excluding _Target_ and _Quintile_) for
    /// a specific timestamp across two datasets. Returns a list of column names
    /// whose values differ.
    /// </summary>
    private static List<string> DiffFeatures(
        List<DatasetRow> rowsA,
        List<DatasetRow> rowsB,
        DateTime timestamp,
        string[] columns)
    {
        var rowA = rowsA.FirstOrDefault(r => r.Timestamp == timestamp);
        var rowB = rowsB.FirstOrDefault(r => r.Timestamp == timestamp);

        Assert.NotNull(rowA);
        Assert.NotNull(rowB);

        var featureCols = columns.Where(c =>
            !c.Contains("_Target_") && !c.Contains("_Quintile_")).ToArray();

        return featureCols
            .Where(c =>
            {
                var a = rowA.Values.TryGetValue(c, out var av) ? av : null;
                var b = rowB.Values.TryGetValue(c, out var bv) ? bv : null;
                if (a is null && b is null) return false;
                if (a is null || b is null) return true;
                return Math.Abs(a.Value - b.Value) > 1e-14;
            })
            .ToList();
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// The target bar is bar[20] for the row at bar[19].
    /// Modifying bar[22] (two bars after the target) must NOT change any
    /// feature value for the row at bar[19].
    /// </summary>
    [Fact]
    public void Features_Unchanged_WhenBarTwoBeyondTargetIsModified()
    {
        var closes = GenerateCloses(30);
        var (rowsA, cols, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(closes), EmptyIndicators());

        // bar 22: two positions after the target bar (bar 20) for row at bar 19
        var modified = (decimal[])closes.Clone();
        modified[22] *= 3m;

        var (rowsB, _, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(modified), EmptyIndicators());

        var testTimestamp = Base.AddHours(19 * 4);
        var changed = DiffFeatures(rowsA, rowsB, testTimestamp, cols);

        Assert.Empty(changed);
    }

    /// <summary>
    /// Same as above but modifying bar[25] — well beyond the target.
    /// </summary>
    [Fact]
    public void Features_Unchanged_WhenFarFutureBarIsModified()
    {
        var closes = GenerateCloses(30);
        var (rowsA, cols, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(closes), EmptyIndicators());

        var modified = (decimal[])closes.Clone();
        modified[25] *= 5m;

        var (rowsB, _, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(modified), EmptyIndicators());

        var testTimestamp = Base.AddHours(19 * 4);
        var changed = DiffFeatures(rowsA, rowsB, testTimestamp, cols);

        Assert.Empty(changed);
    }

    /// <summary>
    /// The target bar is bar[20]. Modifying bar[20] must change the target
    /// value but leave every feature column unchanged.
    /// </summary>
    [Fact]
    public void Target_Changes_ButFeatures_Unchanged_WhenTargetBarIsModified()
    {
        var closes = GenerateCloses(30);
        var (rowsA, cols, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(closes), EmptyIndicators());

        var modified = (decimal[])closes.Clone();
        modified[20] *= 1.05m;   // 5% shift in the target bar close

        var (rowsB, _, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(modified), EmptyIndicators());

        var testTimestamp = Base.AddHours(19 * 4);

        // ── Features must be identical ─────────────────────────────────────
        var changedFeatures = DiffFeatures(rowsA, rowsB, testTimestamp, cols);
        Assert.Empty(changedFeatures);

        // ── Target must have changed ────────────────────────────────────────
        var targetCol = "XAUUSD_Target_4h_Return";
        var rowA = rowsA.First(r => r.Timestamp == testTimestamp);
        var rowB = rowsB.First(r => r.Timestamp == testTimestamp);

        Assert.NotNull(rowA.Values[targetCol]);
        Assert.NotNull(rowB.Values[targetCol]);
        Assert.NotEqual(rowA.Values[targetCol]!.Value, rowB.Values[targetCol]!.Value,
            precision: 10);
    }

    /// <summary>
    /// TargetVolScalar at bar[19] uses only closes from bars 0–19.
    /// Modifying bar[20] (the target bar) must NOT change TargetVolScalar.
    /// </summary>
    [Fact]
    public void TargetVolScalar_Unchanged_WhenTargetBarIsModified()
    {
        var closes = GenerateCloses(30);
        var (rowsA, _, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(closes), EmptyIndicators());

        var modified = (decimal[])closes.Clone();
        modified[20] *= 10m;   // large change to make any leakage obvious

        var (rowsB, _, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(modified), EmptyIndicators());

        var testTimestamp = Base.AddHours(19 * 4);
        var scalarCol = "XAUUSD_TargetVolScalar";

        var rowA = rowsA.First(r => r.Timestamp == testTimestamp);
        var rowB = rowsB.First(r => r.Timestamp == testTimestamp);

        Assert.NotNull(rowA.Values[scalarCol]);
        Assert.NotNull(rowB.Values[scalarCol]);
        Assert.Equal(rowA.Values[scalarCol]!.Value, rowB.Values[scalarCol]!.Value,
            precision: 12);
    }

    // ── Cross-asset ratio leakage tests ──────────────────────────────────────
    //
    // The ratios are now z-scored with a 252-bar rolling window (= AdZScorePeriod
    // default).  Bar index 251 is the first bar with a valid z-score value, so
    // all cross-asset tests use 258 bars and probe the row at bar[251].
    // Bar[252] is the target bar for that row.

    private const int ZBars    = 258;   // total candles — enough for 252-bar warmup + target
    private const int ZTestBar = 251;   // first bar with a valid z-score (period-1 = 251)
    private const int ZTargBar = 252;   // target bar for the row at ZTestBar

    /// <summary>
    /// GoldSilverRatio (z-scored) at bar[251] must not change when bar[252]
    /// (the target bar) of XAG or WTI is modified.
    /// GoldOilRatio (z-scored) must also be unchanged.
    /// </summary>
    [Fact]
    public void CrossAssetRatios_Unchanged_WhenTargetBarOfCounterAssetIsModified()
    {
        var xauCloses = GenerateCloses(ZBars);
        var xagCloses = xauCloses.Select(c => c * 0.015m).ToArray();
        var wtiCloses = xauCloses.Select(c => c * 0.04m).ToArray();

        var candlesA = MakeCrossAssetCandles(xauCloses, xagCloses, wtiCloses);
        var (rowsA, _, _) = Transformer.BuildFeatureMatrix(LeakageCfg(), candlesA, EmptyIndicators());

        var xagMod = (decimal[])xagCloses.Clone(); xagMod[ZTargBar] *= 3m;
        var wtiMod = (decimal[])wtiCloses.Clone(); wtiMod[ZTargBar] *= 3m;

        var candlesB = MakeCrossAssetCandles(xauCloses, xagMod, wtiMod);
        var (rowsB, _, _) = Transformer.BuildFeatureMatrix(LeakageCfg(), candlesB, EmptyIndicators());

        var ts   = Base.AddHours(ZTestBar * 4);
        var rowA = rowsA.First(r => r.Timestamp == ts);
        var rowB = rowsB.First(r => r.Timestamp == ts);

        // Values must be non-null (confirms z-score warmup is satisfied)
        Assert.NotNull(rowA.Values["GoldSilverRatio"]);
        Assert.NotNull(rowA.Values["GoldOilRatio"]);

        Assert.Equal(rowA.Values["GoldSilverRatio"], rowB.Values["GoldSilverRatio"]);
        Assert.Equal(rowA.Values["GoldOilRatio"],    rowB.Values["GoldOilRatio"]);
    }

    /// <summary>
    /// Same guarantee for a bar well beyond the target (bar[255]).
    /// </summary>
    [Fact]
    public void CrossAssetRatios_Unchanged_WhenFarFutureBarOfCounterAssetIsModified()
    {
        var xauCloses = GenerateCloses(ZBars);
        var xagCloses = xauCloses.Select(c => c * 0.015m).ToArray();
        var wtiCloses = xauCloses.Select(c => c * 0.04m).ToArray();

        var candlesA = MakeCrossAssetCandles(xauCloses, xagCloses, wtiCloses);
        var (rowsA, _, _) = Transformer.BuildFeatureMatrix(LeakageCfg(), candlesA, EmptyIndicators());

        var xagMod = (decimal[])xagCloses.Clone(); xagMod[ZTargBar + 3] *= 5m;
        var wtiMod = (decimal[])wtiCloses.Clone(); wtiMod[ZTargBar + 3] *= 5m;

        var candlesB = MakeCrossAssetCandles(xauCloses, xagMod, wtiMod);
        var (rowsB, _, _) = Transformer.BuildFeatureMatrix(LeakageCfg(), candlesB, EmptyIndicators());

        var ts   = Base.AddHours(ZTestBar * 4);
        var rowA = rowsA.First(r => r.Timestamp == ts);
        var rowB = rowsB.First(r => r.Timestamp == ts);

        Assert.NotNull(rowA.Values["GoldSilverRatio"]);
        Assert.NotNull(rowA.Values["GoldOilRatio"]);

        Assert.Equal(rowA.Values["GoldSilverRatio"], rowB.Values["GoldSilverRatio"]);
        Assert.Equal(rowA.Values["GoldOilRatio"],    rowB.Values["GoldOilRatio"]);
    }

    /// <summary>
    /// Sensitivity check: modifying bar[251] of XAG/WTI (the bar whose close
    /// is used in the z-score window for the row at bar[251]) must shift both
    /// ratios, confirming the no-leakage tests above would catch real leakage.
    /// </summary>
    [Fact]
    public void CrossAssetRatios_Sensitive_ToPastBarOfCounterAsset()
    {
        var xauCloses = GenerateCloses(ZBars);
        var xagCloses = xauCloses.Select(c => c * 0.015m).ToArray();
        var wtiCloses = xauCloses.Select(c => c * 0.04m).ToArray();

        var candlesA = MakeCrossAssetCandles(xauCloses, xagCloses, wtiCloses);
        var (rowsA, _, _) = Transformer.BuildFeatureMatrix(LeakageCfg(), candlesA, EmptyIndicators());

        // Halve silver/oil at the test bar itself — must shift the z-scored ratio
        var xagMod = (decimal[])xagCloses.Clone(); xagMod[ZTestBar] *= 0.5m;
        var wtiMod = (decimal[])wtiCloses.Clone(); wtiMod[ZTestBar] *= 0.5m;

        var candlesB = MakeCrossAssetCandles(xauCloses, xagMod, wtiMod);
        var (rowsB, _, _) = Transformer.BuildFeatureMatrix(LeakageCfg(), candlesB, EmptyIndicators());

        var ts   = Base.AddHours(ZTestBar * 4);
        var rowA = rowsA.First(r => r.Timestamp == ts);
        var rowB = rowsB.First(r => r.Timestamp == ts);

        Assert.NotNull(rowA.Values["GoldSilverRatio"]);
        Assert.NotNull(rowB.Values["GoldSilverRatio"]);
        Assert.NotEqual(rowA.Values["GoldSilverRatio"]!.Value, rowB.Values["GoldSilverRatio"]!.Value, precision: 10);
        Assert.NotEqual(rowA.Values["GoldOilRatio"]!.Value,    rowB.Values["GoldOilRatio"]!.Value,    precision: 10);
    }

    /// <summary>
    /// EMA at bar[19] uses closes from bars 0–19. Modifying bar[15] (within
    /// the EMA lookback) must change the EMA feature for the row at bar[19].
    /// </summary>
    [Fact]
    public void Features_Sensitive_ToPastBarWithinLookback()
    {
        var closes = GenerateCloses(30);
        var (rowsA, cols, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(closes), EmptyIndicators());

        var modified = (decimal[])closes.Clone();
        modified[15] *= 1.10m;   // 10% shock to bar 15 (inside EMA-5 window at bar 19)

        var (rowsB, _, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(modified), EmptyIndicators());

        var testTimestamp = Base.AddHours(19 * 4);
        var changed = DiffFeatures(rowsA, rowsB, testTimestamp, cols);

        // At least one feature must have changed
        Assert.NotEmpty(changed);
    }

    /// <summary>
    /// Verifies the exact normalised target formula:
    ///   Target_4h_Return = log(close[20] / close[19]) / max(RealizedVol_20, 1e-8)
    /// where RealizedVol_20 is the population std of log returns over bars 0–19.
    /// </summary>
    [Fact]
    public void Target_Equals_VolNormalisedLogReturn()
    {
        var closes = GenerateCloses(30);
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(closes), EmptyIndicators());

        var testTimestamp = Base.AddHours(19 * 4);
        var row = rows.First(r => r.Timestamp == testTimestamp);

        // Manually compute realized vol over bars 0–19 (log returns 1–19)
        var logRets = Enumerable.Range(1, 19)
            .Select(i => Math.Log((double)closes[i] / (double)closes[i - 1]))
            .ToArray();
        double mean    = logRets.Average();
        double variance = logRets.Average(v => (v - mean) * (v - mean));
        double realizedVol = Math.Sqrt(variance);
        double volScalar   = Math.Max(realizedVol, 1e-8);

        double rawReturn = Math.Log((double)closes[20] / (double)closes[19]);
        double expectedTarget = rawReturn / volScalar;

        Assert.NotNull(row.Values["XAUUSD_Target_4h_Return"]);
        Assert.Equal(expectedTarget, row.Values["XAUUSD_Target_4h_Return"]!.Value,
            precision: 10);
    }

    /// <summary>
    /// Rows before the vol warm-up (bars 0–18) must have a null target so the
    /// Predictor skips them during training — they must NOT leak future vol
    /// into a normalised target.
    /// </summary>
    [Fact]
    public void Rows_BeforeVolWarmup_HaveNullTarget()
    {
        var closes = GenerateCloses(30);
        var (rows, _, _) = Transformer.BuildFeatureMatrix(
            LeakageCfg(), MakeCandles(closes), EmptyIndicators());

        for (int bar = 0; bar < 19; bar++)
        {
            var ts  = Base.AddHours(bar * 4);
            var row = rows.FirstOrDefault(r => r.Timestamp == ts);
            if (row is null) continue;   // row may be skipped entirely

            var target = row.Values["XAUUSD_Target_4h_Return"];
            Assert.Null(target);
        }
    }
}
