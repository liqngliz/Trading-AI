using System.Globalization;
using System.Text;
using System.Text.Json;
using Indicators;
using Integrations.TwelveData;

namespace Importer;

// ── Configuration ─────────────────────────────────────────────────────────────

/// <summary>Configuration that drives the ML dataset build.</summary>
public sealed class DatasetConfig
{
    public static readonly string[] AllAssets =
    [
    // Precious metals
    "XAU/USD", "XAG/USD", "XPT/USD", "XPD/USD",
    // Forex
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "USD/CNH",
    // ETFs
    // Removed (short history): "XMTH", "SUOD", "ZGLD", "WORLD"
    "SHY", "IEF", "TLT", "SPY", "EEM", "ZSL", "DGZ", "NDAQ", "QQQ", "IYY",
    // Oil ETFs
    // Removed (short history on 1day/1week, binding training start): "3SOI"
    // Removed (UK-listed ETF, no intraday data before 2022, causes 249 fully-NaN cols in early folds): "LOIL"
    // Commodities
    // Removed (short history): "CL1", "C_1", "NG/USD"
    // Removed (US-hours-only futures, 73-86% NaN on 4h): "HG1"
    "WTI/USD",
    // Indices
    // Removed (started Aug 2022, was binding training start to 3.5 years): "SVIX"
    "VIXY", "UDN", "UUP"
    ];

    public static readonly string[] ShortTimeframes = ["15min", "30min", "1h"];
    public static readonly string[] LongTimeframes   = ["4h", "1day", "1week"];

    /// <summary>Symbol to predict. Must exist in <see cref="AllAssets"/>.</summary>
    public string TargetSymbol { get; init; } = "XAU/USD";

    /// <summary>Horizons as (timeframe, barsAhead) pairs.</summary>
    public List<(string Tf, int BarsAhead)> TargetHorizons { get; init; } =
        [("4h", 1), ("1day", 1), ("1week", 1)];

    /// <summary>Aggregation statistics applied over short-TF windows.</summary>
    public string[] AggStats { get; init; } = ["mean", "min", "max", "last", "std"];

    // Indicator periods — 5 per indicator: shortest, short, medium, long, longest
    public int[] SmaPeriods       { get; init; } = [];
    public int[] EmaPeriods       { get; init; } = [24, 50];
    public int[] RsiPeriods       { get; init; } = [24];
    public int[] RocPeriods       { get; init; } = [];
    public int[] StdDevPeriods    { get; init; } = [];
    public int[] AtrPeriods       { get; init; } = [14];
    public int[] BbPeriods        { get; init; } = [20];
    /// <summary>Lookback for rolling z-score of BB bandwidth (compression detector). 0 = disabled.</summary>
    public int   BbWidthZPeriod   { get; init; } = 100;
    public int[] CciPeriods       { get; init; } = [14];
    public int[] WilliamsRPeriods { get; init; } = [];
    public int[] StochKPeriods    { get; init; } = [];
    public int[] AdxPeriods       { get; init; } = [14];
    public int StochDPeriod       { get; init; } = 0;
    public int MacdFast           { get; init; } = 12;
    public int MacdSlow           { get; init; } = 26;
    public int MacdSignal         { get; init; } = 9;
    public int DistN              { get; init; } = 20;
    public int AdZScorePeriod     { get; init; } = 252;

    /// <summary>Window (bars) for rolling std of log-returns on target symbol 4h series. 0 = disabled.</summary>
    public int RealizedVolPeriod  { get; init; } = 20;
    /// <summary>Window (bars) for rolling mean of realized vol used in the vol ratio. 0 = disabled.</summary>
    public int VolRatioMaPeriod   { get; init; } = 200;

    public int WalkForwardFolds { get; init; } = 10;
    public int PurgeBarsGap     { get; init; } = 50;  // in 4h bars (~200 hours; prevents regime autocorrelation leaking across the train/val boundary)

    /// <summary>
    /// Earliest timestamp to include in the dataset (inclusive).
    /// Rows before this date are skipped. Typically set to the 4h commonStart
    /// — the latest first real (non-filled) candle across all symbols — so
    /// all assets have genuine data throughout the training window.
    /// </summary>
    public DateTime? TrainingStartDate { get; init; }

    public string OutputDirectory { get; init; } = "Dataset";
}

// ── Row ───────────────────────────────────────────────────────────────────────

/// <summary>A single assembled sample row.</summary>
public sealed class DatasetRow
{
    public DateTime Timestamp { get; init; }
    /// <summary>Feature + target values keyed by column name.</summary>
    public Dictionary<string, double?> Values { get; init; } = [];
}

// ── Transformer ───────────────────────────────────────────────────────────────

public static class Transformer
{
    private static readonly JsonSerializerOptions JsonIndented = new() { WriteIndented = true };

    private static readonly string[] CrossAssetCols =
    [
        "GoldSilverRatio", "GoldOilRatio",
        "DXY_Mom_1h", "DXY_Mom_4h",
        "YieldCurveProxy", "RiskSentiment", "VolatilityRegime"
    ];

    private static readonly string[] TimeCols =
    [
        "HourOfDay_Sin", "HourOfDay_Cos",
        "DayOfWeek_Sin", "DayOfWeek_Cos",
        "Session"
    ];

    // ── Pre-computed data per (symbol, interval) ──────────────────────────────

    private sealed class SeriesData
    {
        public DateTime[] Times   { get; init; } = [];
        public double[]   Close   { get; init; } = [];
        // Indicators[i][j]: indicator i, bar j — same length as Times
        public double[][] Indicators { get; init; } = [];
    }

    // ── Public Methods ────────────────────────────────────────────────────────

    /// <summary>
    /// Assembles the full multi-timeframe feature matrix for all valid 4h timestamps.
    /// </summary>
    /// <param name="cfg">Dataset configuration.</param>
    /// <param name="allCandles">symbol → interval → time-sorted candles.</param>
    /// <param name="allIndicators">symbol → "endpoint/interval" → time-sorted indicator values.</param>
    /// <returns>Assembled rows and the column header array.</returns>
    public static (List<DatasetRow> Rows, string[] Columns, Dictionary<string, (int NullCount, int EmptyCount, int ZeroCount)> ColumnStats) BuildFeatureMatrix(
        DatasetConfig cfg,
        IReadOnlyDictionary<string, Dictionary<string, List<TimeSeriesValue>>> allCandles,
        IReadOnlyDictionary<string, Dictionary<string, List<IndicatorValue>>> allIndicators)
    {
        if (!DatasetConfig.AllAssets.Contains(cfg.TargetSymbol, StringComparer.OrdinalIgnoreCase))
            throw new ArgumentException(
                $"TargetSymbol '{cfg.TargetSymbol}' not found in assets list. " +
                $"Valid assets: {string.Join(", ", DatasetConfig.AllAssets)}",
                nameof(cfg));

        // ── Pre-compute all series data ───────────────────────────────────────

        // shortSd[assetIdx][tfIdx] / longSd[assetIdx][tfIdx] — null when no data
        var shortSd = new SeriesData?[DatasetConfig.AllAssets.Length][];
        var longSd  = new SeriesData?[DatasetConfig.AllAssets.Length][];
        for (int ai = 0; ai < DatasetConfig.AllAssets.Length; ai++)
        {
            shortSd[ai] = new SeriesData?[DatasetConfig.ShortTimeframes.Length];
            longSd[ai]  = new SeriesData?[DatasetConfig.LongTimeframes.Length];
            var sym = DatasetConfig.AllAssets[ai];
            if (!allCandles.TryGetValue(sym, out var symCandles)) continue;
            allIndicators.TryGetValue(sym, out var symInds);

            for (int ti = 0; ti < DatasetConfig.ShortTimeframes.Length; ti++)
            {
                var tf = DatasetConfig.ShortTimeframes[ti];
                if (symCandles.TryGetValue(tf, out var c) && c.Count > 0)
                    shortSd[ai][ti] = BuildSeriesData(sym, tf, c, symInds, cfg);
            }
            for (int ti = 0; ti < DatasetConfig.LongTimeframes.Length; ti++)
            {
                var tf = DatasetConfig.LongTimeframes[ti];
                if (symCandles.TryGetValue(tf, out var c) && c.Count > 0)
                    longSd[ai][ti] = BuildSeriesData(sym, tf, c, symInds, cfg);
            }
        }

        // ── Build column registry ─────────────────────────────────────────────

        var columns   = new List<string>(2048);
        var colIndex  = new Dictionary<string, int>(2048);
        void AddCol(string name) { colIndex[name] = columns.Count; columns.Add(name); }

        AddCol("Timestamp");
        foreach (var c in TimeCols)        AddCol(c);
        foreach (var c in CrossAssetCols)  AddCol(c);

        var indicatorNames = BuildIndicatorNames(cfg);
        int nInds  = indicatorNames.Length;
        int nStats = cfg.AggStats.Length;

        for (int ai = 0; ai < DatasetConfig.AllAssets.Length; ai++)
        {
            var safe = SafeSymbol(DatasetConfig.AllAssets[ai]);
            for (int ti = 0; ti < DatasetConfig.ShortTimeframes.Length; ti++)
            {
                if (shortSd[ai][ti] == null) continue;
                var tf = DatasetConfig.ShortTimeframes[ti];
                foreach (var ind in indicatorNames)
                    foreach (var stat in cfg.AggStats)
                        AddCol($"{safe}_{tf}_{ind}_{stat}");
            }
            AddCol($"{safe}_Missing");
            for (int ti = 0; ti < DatasetConfig.LongTimeframes.Length; ti++)
            {
                if (longSd[ai][ti] == null) continue;
                var tf = DatasetConfig.LongTimeframes[ti];
                foreach (var ind in indicatorNames)
                    AddCol($"{safe}_{tf}_{ind}");
            }
        }

        var targetSafe = SafeSymbol(cfg.TargetSymbol);
        if (cfg.RealizedVolPeriod > 0)
            AddCol($"{targetSafe}_RealizedVol_{cfg.RealizedVolPeriod}");
        if (cfg.RealizedVolPeriod > 0 && cfg.VolRatioMaPeriod > 0)
            AddCol($"{targetSafe}_VolRatio_{cfg.RealizedVolPeriod}_{cfg.VolRatioMaPeriod}");
        if (cfg.RealizedVolPeriod > 0)
            AddCol($"{targetSafe}_TargetVolScalar");
        foreach (var (tf, _) in cfg.TargetHorizons) AddCol($"{targetSafe}_Target_{tf}_Return");
        foreach (var (tf, _) in cfg.TargetHorizons) AddCol($"{targetSafe}_Target_{tf}_Quintile");

        int nCols = columns.Count;

        // ── Pre-compute column index arrays (avoid dict lookups in hot loop) ──

        // shortIdx[ai][ti][ii][si] = column index, or -1
        var shortIdx = new int[DatasetConfig.AllAssets.Length][][][];
        for (int ai = 0; ai < DatasetConfig.AllAssets.Length; ai++)
        {
            shortIdx[ai] = new int[DatasetConfig.ShortTimeframes.Length][][];
            var safe = SafeSymbol(DatasetConfig.AllAssets[ai]);
            for (int ti = 0; ti < DatasetConfig.ShortTimeframes.Length; ti++)
            {
                shortIdx[ai][ti] = new int[nInds][];
                var tf = DatasetConfig.ShortTimeframes[ti];
                for (int ii = 0; ii < nInds; ii++)
                {
                    shortIdx[ai][ti][ii] = new int[nStats];
                    for (int si = 0; si < nStats; si++)
                    {
                        var key = $"{safe}_{tf}_{indicatorNames[ii]}_{cfg.AggStats[si]}";
                        shortIdx[ai][ti][ii][si] = colIndex.TryGetValue(key, out var ci) ? ci : -1;
                    }
                }
            }
        }

        // longIdx[ai][ti][ii] = column index, or -1
        var longIdx = new int[DatasetConfig.AllAssets.Length][][];
        for (int ai = 0; ai < DatasetConfig.AllAssets.Length; ai++)
        {
            longIdx[ai] = new int[DatasetConfig.LongTimeframes.Length][];
            var safe = SafeSymbol(DatasetConfig.AllAssets[ai]);
            for (int ti = 0; ti < DatasetConfig.LongTimeframes.Length; ti++)
            {
                longIdx[ai][ti] = new int[nInds];
                var tf = DatasetConfig.LongTimeframes[ti];
                for (int ii = 0; ii < nInds; ii++)
                {
                    var key = $"{safe}_{tf}_{indicatorNames[ii]}";
                    longIdx[ai][ti][ii] = colIndex.TryGetValue(key, out var ci) ? ci : -1;
                }
            }
        }

        // Missing-flag and time column indices
        var missingIdx = new int[DatasetConfig.AllAssets.Length];
        for (int ai = 0; ai < DatasetConfig.AllAssets.Length; ai++)
        {
            var key = $"{SafeSymbol(DatasetConfig.AllAssets[ai])}_Missing";
            missingIdx[ai] = colIndex.TryGetValue(key, out var ci) ? ci : -1;
        }

        // Target asset index
        int targetAi = Array.IndexOf(DatasetConfig.AllAssets, cfg.TargetSymbol);

        // ── Locate 4h base bars for target asset ──────────────────────────────

        int base4hTi = Array.IndexOf(DatasetConfig.LongTimeframes, "4h");
        var baseSd   = targetAi >= 0 && base4hTi >= 0 ? longSd[targetAi][base4hTi] : null;

        if (baseSd == null || baseSd.Times.Length == 0)
        {
            Console.WriteLine($"[BuildFeatureMatrix] No 4h data for {cfg.TargetSymbol}. Aborting.");
            return ([], columns.ToArray(), []);
        }

        // ── Pre-compute realized vol and vol ratio for target symbol (4h) ─────
        int nBars = baseSd.Times.Length;
        var realizedVols = new double?[nBars];
        var volRatios    = new double?[nBars];

        if (cfg.RealizedVolPeriod > 0)
        {
            var logRets = new double?[nBars];
            for (int i = 1; i < nBars; i++)
            {
                double c0 = baseSd.Close[i - 1], c1 = baseSd.Close[i];
                if (c0 > 0) logRets[i] = Math.Log(c1 / c0);
            }

            for (int i = cfg.RealizedVolPeriod - 1; i < nBars; i++)
            {
                double sum = 0, sumSq = 0; int cnt = 0;
                for (int j = i - cfg.RealizedVolPeriod + 1; j <= i; j++)
                {
                    if (!logRets[j].HasValue) continue;
                    sum += logRets[j]!.Value; sumSq += logRets[j]!.Value * logRets[j]!.Value; cnt++;
                }
                if (cnt >= 2)
                {
                    double mean = sum / cnt;
                    realizedVols[i] = Math.Sqrt(Math.Max(0, sumSq / cnt - mean * mean));
                }
            }

            if (cfg.VolRatioMaPeriod > 0)
            {
                for (int i = cfg.VolRatioMaPeriod - 1; i < nBars; i++)
                {
                    if (!realizedVols[i].HasValue) continue;
                    double maSum = 0; int maCnt = 0;
                    for (int j = i - cfg.VolRatioMaPeriod + 1; j <= i; j++)
                    {
                        if (realizedVols[j].HasValue) { maSum += realizedVols[j]!.Value; maCnt++; }
                    }
                    if (maCnt > 0 && maSum > 0)
                        volRatios[i] = realizedVols[i]!.Value / (maSum / maCnt);
                }
            }
        }

        // Pre-look up cross-asset SeriesData indices
        int xauAi  = Array.IndexOf(DatasetConfig.AllAssets, "XAU/USD");
        int xagAi  = Array.IndexOf(DatasetConfig.AllAssets, "XAG/USD");
        int wtiAi  = Array.IndexOf(DatasetConfig.AllAssets, "WTI/USD");
        int tltAi  = Array.IndexOf(DatasetConfig.AllAssets, "TLT");
        int shyAi  = Array.IndexOf(DatasetConfig.AllAssets, "SHY");
        int spyAi  = Array.IndexOf(DatasetConfig.AllAssets, "SPY");
        int vixyAi = Array.IndexOf(DatasetConfig.AllAssets, "VIXY");
        int udnAi  = Array.IndexOf(DatasetConfig.AllAssets, "UDN");
        int tf1hShortTi = Array.IndexOf(DatasetConfig.ShortTimeframes, "1h");
        // cross-asset DXY mom uses 1h short and 4h long UDN

        // ── Pre-compute cross-asset ratio z-scores ────────────────────────────
        // Raw price ratios (gold/silver, gold/oil) are non-stationary: as gold
        // trends the level drifts, letting the model learn which regime/year it
        // is in rather than a genuine cross-asset signal.  Z-scoring over a
        // rolling window removes the level before the model sees the values.
        var sdXau4h = xauAi >= 0 && base4hTi >= 0 ? longSd[xauAi][base4hTi] : null;
        var sdXag4h = xagAi >= 0 && base4hTi >= 0 ? longSd[xagAi][base4hTi] : null;
        var sdWti4h = wtiAi >= 0 && base4hTi >= 0 ? longSd[wtiAi][base4hTi] : null;
        var gsRatioRaw = new double[nBars];
        var goRatioRaw = new double[nBars];
        for (int i = 0; i < nBars; i++)
        {
            var t       = baseSd.Times[i];
            int gi      = sdXau4h != null ? BsFloor(sdXau4h.Times, t) : -1;
            int agi     = sdXag4h != null ? BsFloor(sdXag4h.Times, t) : -1;
            int wi      = sdWti4h != null ? BsFloor(sdWti4h.Times, t) : -1;
            double gold   = gi  >= 0 ? sdXau4h!.Close[gi]  : double.NaN;
            double silver = agi >= 0 ? sdXag4h!.Close[agi] : double.NaN;
            double oil    = wi  >= 0 ? sdWti4h!.Close[wi]  : double.NaN;
            gsRatioRaw[i] = silver > 0 ? gold / silver : double.NaN;
            goRatioRaw[i] = oil    > 0 ? gold / oil    : double.NaN;
        }
        int ratioZPeriod   = cfg.AdZScorePeriod > 0 ? cfg.AdZScorePeriod : 252;
        double[] goldSilverZ = RollingZScore(gsRatioRaw, ratioZPeriod);
        double[] goldOilZ    = RollingZScore(goRatioRaw, ratioZPeriod);

        // Pre-computed column indices for time + cross-asset
        int ciHourSin = colIndex["HourOfDay_Sin"], ciHourCos = colIndex["HourOfDay_Cos"];
        int ciDowSin  = colIndex["DayOfWeek_Sin"],  ciDowCos  = colIndex["DayOfWeek_Cos"];
        int ciSession = colIndex["Session"];
        var xaCols    = CrossAssetCols.Select(c => colIndex.TryGetValue(c, out var i) ? i : -1).ToArray();

        // Indicator indices used by cross-asset features
        int indLogReturn = 0; // always first
        int indRsi = cfg.RsiPeriods.Length > 0
            ? Array.IndexOf(indicatorNames, cfg.RsiPeriods.Contains(14) ? "RSI_14" : $"RSI_{cfg.RsiPeriods[0]}")
            : -1;
        int indAtr = cfg.AtrPeriods.Length > 0
            ? Array.IndexOf(indicatorNames, cfg.AtrPeriods.Contains(14) ? "ATR_14" : $"ATR_{cfg.AtrPeriods[0]}")
            : -1;

        // ── Main loop ─────────────────────────────────────────────────────────

        var cols        = columns.ToArray();
        var rows        = new List<DatasetRow>(baseSd.Times.Length);
        int skipped     = 0;
        var nullCounts  = new int[nCols];
        var zeroCounts  = new int[nCols]; // strictly 0.0 (non-null)
        var emptyCounts = new int[nCols]; // null + zero combined

        for (int bi = 0; bi < baseSd.Times.Length; bi++)
        {
            var T = baseSd.Times[bi];
            if (cfg.TrainingStartDate.HasValue && T < cfg.TrainingStartDate.Value) continue;

            var windowStart = T.AddHours(-4);
            var vals        = new double?[nCols];

            // ── Time encodings ────────────────────────────────────────────────
            double hour = T.Hour + T.Minute / 60.0;
            vals[ciHourSin] = Math.Sin(2 * Math.PI * hour / 24.0);
            vals[ciHourCos] = Math.Cos(2 * Math.PI * hour / 24.0);
            double dow = (double)T.DayOfWeek;
            vals[ciDowSin]  = Math.Sin(2 * Math.PI * dow / 7.0);
            vals[ciDowCos]  = Math.Cos(2 * Math.PI * dow / 7.0);
            vals[ciSession] = (double)GetSession(T);

            // ── Per-asset features ────────────────────────────────────────────
            for (int ai = 0; ai < DatasetConfig.AllAssets.Length; ai++)
            {
                bool hasMissing = false;

                // Short TF aggregation
                for (int ti = 0; ti < DatasetConfig.ShortTimeframes.Length; ti++)
                {
                    var sd = shortSd[ai][ti];
                    if (sd == null) { hasMissing = true; continue; }

                    int hi = BsFloor(sd.Times, T);
                    if (hi < 0) { hasMissing = true; continue; }

                    // Advance lo to first bar >= windowStart
                    int lo = BsFloor(sd.Times, windowStart);
                    if (lo < 0) lo = 0;
                    while (lo < hi && sd.Times[lo] < windowStart) lo++;

                    WriteShortAgg(vals, sd, lo, hi, shortIdx[ai][ti], nInds, nStats, cfg.AggStats);
                }

                if (missingIdx[ai] >= 0) vals[missingIdx[ai]] = hasMissing ? 1.0 : 0.0;

                // Long TF single bar
                for (int ti = 0; ti < DatasetConfig.LongTimeframes.Length; ti++)
                {
                    var sd = longSd[ai][ti];
                    if (sd == null) continue;
                    int i = BsFloor(sd.Times, T);
                    if (i < 0) continue;
                    WriteLong(vals, sd, i, longIdx[ai][ti], nInds);
                }
            }

            // ── Cross-asset features ──────────────────────────────────────────
            WriteCrossAsset(vals, T, longSd, shortSd,
                goldSilverZ[bi], goldOilZ[bi],
                tltAi, shyAi, spyAi, vixyAi, udnAi,
                base4hTi, tf1hShortTi, xaCols,
                indLogReturn, indRsi, indAtr);

            // ── Realized vol / vol ratio / vol scalar ─────────────────────────
            if (cfg.RealizedVolPeriod > 0 &&
                colIndex.TryGetValue($"{targetSafe}_RealizedVol_{cfg.RealizedVolPeriod}", out var rvColIdx))
                vals[rvColIdx] = realizedVols[bi];

            if (cfg.RealizedVolPeriod > 0 && cfg.VolRatioMaPeriod > 0 &&
                colIndex.TryGetValue($"{targetSafe}_VolRatio_{cfg.RealizedVolPeriod}_{cfg.VolRatioMaPeriod}", out var vrColIdx))
                vals[vrColIdx] = volRatios[bi];

            double? volScalar = null;
            if (cfg.RealizedVolPeriod > 0 && realizedVols[bi].HasValue)
            {
                volScalar = Math.Max(realizedVols[bi]!.Value, 1e-8);
                if (colIndex.TryGetValue($"{targetSafe}_TargetVolScalar", out var vsColIdx))
                    vals[vsColIdx] = volScalar;
            }

            // ── Targets ───────────────────────────────────────────────────────
            bool hasTarget = false;
            if (baseSd.Times[bi] == T)
            {
                hasTarget = true;
                for (int hi = 0; hi < cfg.TargetHorizons.Count; hi++)
                {
                    var (tf, barsAhead) = cfg.TargetHorizons[hi];
                    var retColKey = $"{targetSafe}_Target_{tf}_Return";
                    if (!colIndex.TryGetValue(retColKey, out var retColIdx)) continue;

                    int tfTi = Array.IndexOf(DatasetConfig.LongTimeframes, tf);
                    if (tfTi < 0 || targetAi < 0) continue;
                    var tsd = longSd[targetAi][tfTi];
                    if (tsd == null) continue;

                    int curIdx    = BsFloor(tsd.Times, T);
                    int futureIdx = curIdx + barsAhead;
                    if (curIdx < 0 || futureIdx >= tsd.Times.Length) continue;

                    double cNow    = tsd.Close[curIdx];
                    double cFuture = tsd.Close[futureIdx];
                    if (cNow > 0)
                    {
                        double rawReturn = Math.Log(cFuture / cNow);
                        if (cfg.RealizedVolPeriod > 0 && volScalar.HasValue)
                            vals[retColIdx] = rawReturn / volScalar.Value;
                        else if (cfg.RealizedVolPeriod == 0)
                            vals[retColIdx] = rawReturn;
                        // else: vol warm-up not met — leave null so row is excluded from training
                    }
                }
            }

            if (!hasTarget) { skipped++; continue; }

            for (int ci = 0; ci < nCols; ci++)
            {
                var v = vals[ci];
                if (!v.HasValue)        { nullCounts[ci]++; emptyCounts[ci]++; }
                else if (v.Value == 0)  { zeroCounts[ci]++; emptyCounts[ci]++; }
            }

            var rowDict = new Dictionary<string, double?>(nCols);
            for (int ci = 0; ci < nCols; ci++) rowDict[cols[ci]] = vals[ci];
            rows.Add(new DatasetRow { Timestamp = T, Values = rowDict });
        }

        var columnStats = new Dictionary<string, (int NullCount, int EmptyCount, int ZeroCount)>(nCols);
        for (int ci = 0; ci < nCols; ci++)
            columnStats[cols[ci]] = (nullCounts[ci], emptyCounts[ci], zeroCounts[ci]);

        Console.WriteLine($"[BuildFeatureMatrix] {rows.Count} rows assembled, {skipped} skipped (no target data).");
        return (rows, cols, columnStats);
    }

    /// <summary>
    /// Removes columns where the fraction of null values exceeds <paramref name="threshold"/> (default 0.8).
    /// Rewrites each row's Values array to only contain kept columns.
    /// Writes the list of pruned column names to <paramref name="reportPath"/>.
    /// </summary>
    public static (List<DatasetRow> Rows, string[] Columns) PruneSparseCols(
        List<DatasetRow> rows,
        string[] columns,
        string reportPath,
        double threshold = 0.8)
    {
        if (rows.Count == 0) return (rows, columns);

        int nCols     = columns.Length;
        int n         = rows.Count;
        var nullCounts = new int[nCols];
        var zeroCounts = new int[nCols];

        foreach (var row in rows)
            for (int ci = 0; ci < nCols; ci++)
            {
                row.Values.TryGetValue(columns[ci], out var v);
                if (!v.HasValue)      { nullCounts[ci]++; zeroCounts[ci]++; }
                else if (v.Value == 0)  zeroCounts[ci]++;
            }

        var keepIdx     = new List<int>(nCols);
        var removedCols = new List<string>();
        for (int ci = 0; ci < nCols; ci++)
        {
            var reason =
                nullCounts[ci] / (double)n > threshold ? $"null={nullCounts[ci] * 100 / n}%" :
                zeroCounts[ci] / (double)n > threshold ? $"null+zero={zeroCounts[ci] * 100 / n}%" :
                null;

            if (reason != null)
                removedCols.Add($"{columns[ci]}  [{reason}]");
            else
                keepIdx.Add(ci);
        }

        if (removedCols.Count == 0)
        {
            Console.WriteLine("[PruneSparseCols] No columns exceeded threshold.");
            return (rows, columns);
        }

        int[] keep = [.. keepIdx];

        var newRows = rows.Select(row =>
        {
            var next = new Dictionary<string, double?>(keep.Length);
            foreach (var ci in keep) next[columns[ci]] = row.Values.GetValueOrDefault(columns[ci]);
            return new DatasetRow { Timestamp = row.Timestamp, Values = next };
        }).ToList();

        var newColumns = keep.Select(i => columns[i]).ToArray();

        Directory.CreateDirectory(Path.GetDirectoryName(reportPath) ?? ".");
        File.WriteAllLines(reportPath, removedCols);

        Console.WriteLine($"[PruneSparseCols] Removed {removedCols.Count} columns (>{threshold:P0} null or null+zero) → {reportPath}");
        return (newRows, newColumns);
    }

    /// <summary>
    /// Exports assembled rows to a CSV file.
    /// Column order: Timestamp → time encodings → cross-asset → per-asset features → targets.
    /// </summary>
    public static void ExportToCsv(List<DatasetRow> rows, string[] columns, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");

        using var sw = new StreamWriter(path, append: false, Encoding.UTF8);

        // Header — columns[0] is "Timestamp" but we write it separately as metadata
        sw.Write("Timestamp");
        for (int i = 1; i < columns.Length; i++) { sw.Write(','); sw.Write(columns[i]); }
        sw.WriteLine();

        var sb = new StringBuilder(columns.Length * 10);
        foreach (var row in rows)
        {
            sb.Clear();
            sb.Append(row.Timestamp.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
            for (int i = 1; i < columns.Length; i++)
            {
                sb.Append(',');
                if (row.Values.TryGetValue(columns[i], out var v) && v.HasValue)
                    sb.Append(v.Value.ToString("G6", CultureInfo.InvariantCulture));
            }
            sw.WriteLine(sb);
        }

        Console.WriteLine($"[ExportToCsv] {rows.Count} rows → {path}");
    }

    /// <summary>
    /// Reads an existing CSV, computes quintile boundaries (20th/40th/60th/80th pctile)
    /// over the full dataset per target return column, appends quintile columns,
    /// and writes a new CSV.
    /// </summary>
    public static void BucketTargets(string csvPath, string outputPath)
    {
        var lines = File.ReadAllLines(csvPath);
        if (lines.Length < 2) return;

        var headers = lines[0].Split(',');
        var targetCols = new List<(int srcIdx, string quintileName)>();
        for (int i = 0; i < headers.Length; i++)
            if (headers[i].EndsWith("_Return", StringComparison.Ordinal))
                targetCols.Add((i, headers[i][..^"_Return".Length] + "_Quintile"));

        if (targetCols.Count == 0) { File.Copy(csvPath, outputPath, overwrite: true); return; }

        int nRows = lines.Length - 1;
        var returnVals = new List<double>[targetCols.Count];
        for (int t = 0; t < targetCols.Count; t++) returnVals[t] = new List<double>(nRows);

        var data = new string[nRows][];
        for (int li = 1; li <= nRows; li++)
        {
            var parts = lines[li].Split(',');
            data[li - 1] = parts;
            for (int t = 0; t < targetCols.Count; t++)
            {
                int src = targetCols[t].srcIdx;
                if (src < parts.Length &&
                    double.TryParse(parts[src], NumberStyles.Any, CultureInfo.InvariantCulture, out var v))
                    returnVals[t].Add(v);
            }
        }

        // Quintile boundaries at 20/40/60/80th percentiles
        var boundaries = new double[targetCols.Count][];
        for (int t = 0; t < targetCols.Count; t++)
        {
            var sorted = returnVals[t].ToArray();
            Array.Sort(sorted);
            boundaries[t] = [Percentile(sorted, 20), Percentile(sorted, 40),
                             Percentile(sorted, 60), Percentile(sorted, 80)];
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
        using var sw = new StreamWriter(outputPath, append: false, Encoding.UTF8);
        sw.Write(lines[0]);
        foreach (var (_, qName) in targetCols) { sw.Write(','); sw.Write(qName); }
        sw.WriteLine();

        var sb = new StringBuilder();
        for (int li = 0; li < nRows; li++)
        {
            sb.Clear();
            sb.Append(string.Join(',', data[li]));
            for (int t = 0; t < targetCols.Count; t++)
            {
                sb.Append(',');
                int src = targetCols[t].srcIdx;
                if (src < data[li].Length &&
                    double.TryParse(data[li][src], NumberStyles.Any, CultureInfo.InvariantCulture, out var v))
                    sb.Append(GetQuintile(v, boundaries[t]));
            }
            sw.WriteLine(sb);
        }

        Console.WriteLine($"[BucketTargets] {nRows} rows with quintile columns → {outputPath}");
    }

    /// <summary>
    /// Creates walk-forward expanding-window splits with a purge gap between train and val.
    /// Exports fold_{n}_train.csv and fold_{n}_val.csv to <see cref="DatasetConfig.OutputDirectory"/>.
    /// </summary>
    public static void CreateWalkForwardSplits(List<DatasetRow> rows, string[] columns, DatasetConfig cfg)
    {
        if (rows.Count == 0) return;
        Directory.CreateDirectory(cfg.OutputDirectory);

        int n       = rows.Count;
        int folds   = cfg.WalkForwardFolds;
        int valSize = n / (folds + 1);

        for (int fold = 0; fold < folds; fold++)
        {
            int trainEnd = (fold + 1) * valSize - 1;
            int valStart = trainEnd + 1 + cfg.PurgeBarsGap;
            int valEnd   = Math.Min(valStart + valSize - 1, n - 1);
            if (valStart > valEnd) continue;

            var trainRows = rows.GetRange(0, trainEnd + 1);
            var valRows   = rows.GetRange(valStart, valEnd - valStart + 1);

            ExportToCsv(trainRows, columns, Path.Combine(cfg.OutputDirectory, $"fold_{fold + 1}_train.csv"));
            ExportToCsv(valRows,   columns, Path.Combine(cfg.OutputDirectory, $"fold_{fold + 1}_val.csv"));

            Console.WriteLine(
                $"[Fold {fold + 1}] Train {trainRows.Count} rows " +
                $"({rows[0].Timestamp:yyyy-MM-dd}→{rows[trainEnd].Timestamp:yyyy-MM-dd})  " +
                $"Val {valRows.Count} rows " +
                $"({rows[valStart].Timestamp:yyyy-MM-dd}→{rows[valEnd].Timestamp:yyyy-MM-dd})");
        }
    }

    /// <summary>
    /// Writes a summary report: row count, date range, missing value rates,
    /// target return statistics, and quintile distributions.
    /// </summary>
    public static void GenerateDatasetReport(List<DatasetRow> rows, string[] columns, string outputPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
        using var sw = new StreamWriter(outputPath, append: false, Encoding.UTF8);

        if (rows.Count == 0) { sw.WriteLine("No rows in dataset."); return; }

        sw.WriteLine("=== Dataset Report ===");
        sw.WriteLine($"Rows:       {rows.Count}");
        sw.WriteLine($"Columns:    {columns.Length}");
        sw.WriteLine($"Date range: {rows[0].Timestamp:yyyy-MM-dd HH:mm} → {rows[^1].Timestamp:yyyy-MM-dd HH:mm}");
        sw.WriteLine();

        sw.WriteLine("--- Missing Value Rates ---");
        var missing = new int[columns.Length];
        foreach (var row in rows)
            for (int i = 0; i < columns.Length; i++)
                if (!row.Values.TryGetValue(columns[i], out var mv) || !mv.HasValue) missing[i]++;
        for (int i = 1; i < columns.Length; i++)
            if (missing[i] > 0)
                sw.WriteLine($"  {columns[i]}: {missing[i]} ({100.0 * missing[i] / rows.Count:F1}%)");

        sw.WriteLine();
        sw.WriteLine("--- Target Return Statistics ---");
        for (int i = 1; i < columns.Length; i++)
        {
            if (!columns[i].Contains("_Target_")) continue;
            int count = 0; double sum = 0, sumSq = 0, mn = double.MaxValue, mx = double.MinValue;
            foreach (var row in rows)
            {
                if (!row.Values.TryGetValue(columns[i], out var rv) || !rv.HasValue) continue;
                var v = rv.Value;
                sum += v; sumSq += v * v; count++;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            if (count == 0) continue;
            double mean = sum / count;
            double std  = Math.Sqrt(Math.Max(0, sumSq / count - mean * mean));
            sw.WriteLine($"  {columns[i]}: n={count} mean={mean:F6} std={std:F6} min={mn:F6} max={mx:F6}");
        }

        Console.WriteLine($"[GenerateDatasetReport] Report → {outputPath}");
    }

    /// <summary>
    /// Scans all rows for null feature values and writes a JSON report grouped by symbol/interval/indicator.
    /// </summary>
    public static void GenerateNullReport(List<DatasetRow> rows, string[] columns, string outputPath)
    {
        if (rows.Count == 0) return;
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");

        int nCols = columns.Length;

        // Per-column: null count + up to 10 sample timestamps
        var nullCounts  = new int[nCols];
        var nullSamples = new List<string>[nCols];
        for (int ci = 0; ci < nCols; ci++) nullSamples[ci] = [];

        foreach (var row in rows)
        {
            for (int ci = 0; ci < nCols; ci++)
            {
                if (!row.Values.TryGetValue(columns[ci], out var v) || !v.HasValue)
                {
                    nullCounts[ci]++;
                    if (nullSamples[ci].Count < 10)
                        nullSamples[ci].Add(row.Timestamp.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));
                }
            }
        }

        // Parse column name into (symbol, interval, indicator, stat)
        // Patterns:
        //   XAUUSD_4h_LogReturn            → long TF
        //   XAUUSD_1h_LogReturn_mean       → short TF agg
        //   XAUUSD_Missing                 → missing flag
        //   XAUUSD_Target_4h_Return        → target
        //   HourOfDay_Sin / Session / ...  → time/cross-asset
        static (string symbol, string interval, string indicator) ParseColumn(string col)
        {
            var knownTfs = new HashSet<string>(["15min","30min","1h","2h","4h","8h","1day","1week"], StringComparer.Ordinal);
            var parts = col.Split('_');
            // Find timeframe token
            for (int p = 1; p < parts.Length; p++)
            {
                if (knownTfs.Contains(parts[p]))
                {
                    var sym = string.Join("_", parts[..p]);
                    var tf  = parts[p];
                    var ind = p + 1 < parts.Length ? parts[p + 1] : "";
                    return (sym, tf, ind);
                }
            }
            return ("", "", col); // time / cross-asset / target
        }

        var columnEntries = new List<object>(nCols);
        for (int ci = 0; ci < nCols; ci++)
        {
            if (nullCounts[ci] == 0) continue;
            var (sym, tf, ind) = ParseColumn(columns[ci]);
            columnEntries.Add(new
            {
                column        = columns[ci],
                symbol        = sym,
                interval      = tf,
                indicator     = ind,
                nullCount     = nullCounts[ci],
                nullPct       = Math.Round(nullCounts[ci] * 100.0 / rows.Count, 2),
                sampleTimestamps = nullSamples[ci]
            });
        }

        // Group by (symbol, interval) for summary
        var grouped = columnEntries
            .Cast<dynamic>()
            .GroupBy(e => $"{e.symbol}|{e.interval}")
            .Select(g =>
            {
                var parts = g.Key.Split('|');
                int totalNulls = g.Sum(e => (int)e.nullCount);
                int totalCells = g.Count() * rows.Count;
                return new
                {
                    symbol       = parts[0],
                    interval     = parts.Length > 1 ? parts[1] : "",
                    columnsWithNulls = g.Count(),
                    totalNullCells   = totalNulls,
                    nullPct          = Math.Round(totalNulls * 100.0 / totalCells, 2)
                };
            })
            .OrderByDescending(x => x.totalNullCells)
            .ToList();

        var report = new
        {
            generatedAt   = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
            totalRows     = rows.Count,
            totalColumns  = nCols,
            columnsWithNulls = columnEntries.Count,
            symbolIntervalSummary = grouped,
            columnDetail  = columnEntries
        };

        var json = JsonSerializer.Serialize(report, JsonIndented);
        File.WriteAllText(outputPath, json);
        Console.WriteLine($"[GenerateNullReport] {columnEntries.Count} columns with nulls → {outputPath}");
    }

    // ── Private: build per-(symbol,interval) data ─────────────────────────────

    private static SeriesData BuildSeriesData(
        string symbol, string tf,
        List<TimeSeriesValue> candles,
        Dictionary<string, List<IndicatorValue>>? symInds,
        DatasetConfig cfg)
    {
        int n = candles.Count;
        var times = new DateTime[n];
        var close = new double[n];
        var open  = new double[n];
        var high  = new double[n];
        var low   = new double[n];

        // Build lookup dict for PriceIndicators
        var dict = new Dictionary<DateTime, TimeSeriesValue>(n);
        for (int i = 0; i < n; i++)
        {
            var c = candles[i];
            times[i] = c.Datetime; close[i] = (double)c.Close;
            open[i]  = (double)c.Open; high[i] = (double)c.High; low[i] = (double)c.Low;
            dict[c.Datetime] = c;
        }

        var indList = new List<double[]>();
        try
        {
            // Pre-compute ATR maps (also used for candle shape normalization)
            var atrMaps = cfg.AtrPeriods.Select(p => PriceIndicators.Atr(dict, p)).ToArray();

            // ── LogReturn (period-independent) ───────────────────────────────
            {
                var map = PriceIndicators.LogReturn(dict);
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = map.TryGetValue(times[i], out var v) ? (double)v : double.NaN;
                indList.Add(arr);
            }

            // ── SMA variants ─────────────────────────────────────────────────
            var smaArrays = new List<double[]>(cfg.SmaPeriods.Length);
            foreach (var p in cfg.SmaPeriods)
            {
                var map = PriceIndicators.Sma(dict, p);
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = map.TryGetValue(times[i], out var v) && close[i] != 0 ? (double)v / close[i] : double.NaN;
                indList.Add(arr);
                smaArrays.Add(arr);
            }

            // ── EMA variants ─────────────────────────────────────────────────
            var emaArrays = new List<double[]>(cfg.EmaPeriods.Length);
            foreach (var p in cfg.EmaPeriods)
            {
                var map = PriceIndicators.Ema(dict, p);
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = map.TryGetValue(times[i], out var v) && close[i] != 0 ? (double)v / close[i] : double.NaN;
                indList.Add(arr);
                emaArrays.Add(arr);
            }

            // ── SMASpread: consecutive period pairs (fast/slow ratio) ─────────
            for (int k = 0; k + 1 < smaArrays.Count; k++)
            {
                var fast = smaArrays[k]; var slow = smaArrays[k + 1];
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = !double.IsNaN(fast[i]) && !double.IsNaN(slow[i]) && slow[i] != 0
                        ? fast[i] / slow[i] : double.NaN;
                indList.Add(arr);
            }

            // ── EMASpread: consecutive period pairs (fast/slow ratio) ─────────
            for (int k = 0; k + 1 < emaArrays.Count; k++)
            {
                var fast = emaArrays[k]; var slow = emaArrays[k + 1];
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = !double.IsNaN(fast[i]) && !double.IsNaN(slow[i]) && slow[i] != 0
                        ? fast[i] / slow[i] : double.NaN;
                indList.Add(arr);
            }

            // ── RSI variants ─────────────────────────────────────────────────
            foreach (var p in cfg.RsiPeriods)
            {
                var map = PriceIndicators.Rsi(dict, p);
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = map.TryGetValue(times[i], out var v) ? (double)v / 50.0 - 1.0 : double.NaN;
                indList.Add(arr);
            }

            // ── ROC variants ─────────────────────────────────────────────────
            foreach (var p in cfg.RocPeriods)
            {
                var map = PriceIndicators.Roc(dict, p);
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = map.TryGetValue(times[i], out var v) ? (double)v : double.NaN;
                indList.Add(arr);
            }

            // ── StdDev variants ───────────────────────────────────────────────
            foreach (var p in cfg.StdDevPeriods)
            {
                var map = PriceIndicators.RollingStdDev(dict, p);
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = map.TryGetValue(times[i], out var v) && close[i] != 0 ? (double)v / Math.Abs(close[i]) : double.NaN;
                indList.Add(arr);
            }

            // ── ATR variants (normalized by close) ────────────────────────────
            foreach (var atrMap in atrMaps)
            {
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                {
                    var raw = atrMap.TryGetValue(times[i], out var av) ? (double)av : double.NaN;
                    arr[i] = !double.IsNaN(raw) && close[i] != 0 ? raw / close[i] : double.NaN;
                }
                indList.Add(arr);
            }

            // ── ADX variants (ADX + +DI + -DI, normalised to [-1,1]) ──────────
            foreach (var p in cfg.AdxPeriods)
            {
                var map = PriceIndicators.Adx(dict, p);
                var adxArr = new double[n]; var plusArr = new double[n]; var minusArr = new double[n];
                for (int i = 0; i < n; i++)
                {
                    if (map.TryGetValue(times[i], out var v))
                    { adxArr[i] = (double)v.Adx / 50.0 - 1.0; plusArr[i] = (double)v.PlusDI / 50.0 - 1.0; minusArr[i] = (double)v.MinusDI / 50.0 - 1.0; }
                    else adxArr[i] = plusArr[i] = minusArr[i] = double.NaN;
                }
                indList.Add(adxArr); indList.Add(plusArr); indList.Add(minusArr);
            }

            // ── BB variants (Upper, Lower, Width per period) ──────────────────
            foreach (var p in cfg.BbPeriods)
            {
                var map = PriceIndicators.BollingerBands(dict, p);
                var upper = new double[n]; var lower = new double[n]; var width = new double[n];
                for (int i = 0; i < n; i++)
                {
                    if (map.TryGetValue(times[i], out var bb) && close[i] != 0)
                    { upper[i] = (double)bb.Upper / close[i]; lower[i] = (double)bb.Lower / close[i]; width[i] = (double)bb.Bandwidth; }
                    else upper[i] = lower[i] = width[i] = double.NaN;
                }
                indList.Add(upper); indList.Add(lower); indList.Add(width);
                if (cfg.BbWidthZPeriod > 0)
                    indList.Add(RollingZScore(width, cfg.BbWidthZPeriod));
            }

            // ── CCI variants ──────────────────────────────────────────────────
            foreach (var p in cfg.CciPeriods)
            {
                var map = PriceIndicators.Cci(dict, p);
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = map.TryGetValue(times[i], out var v) ? Math.Max(-1.0, Math.Min(1.0, (double)v / 300.0)) : double.NaN;
                indList.Add(arr);
            }

            // ── WilliamsR variants ────────────────────────────────────────────
            foreach (var p in cfg.WilliamsRPeriods)
            {
                var map = PriceIndicators.WilliamsR(dict, p);
                var arr = new double[n];
                for (int i = 0; i < n; i++)
                    arr[i] = map.TryGetValue(times[i], out var v) ? (double)v / 50.0 + 1.0 : double.NaN;
                indList.Add(arr);
            }

            // ── Stochastic variants (K + D pair per K-period) ─────────────────
            foreach (var p in cfg.StochKPeriods)
            {
                var map = PriceIndicators.Stochastic(dict, p, cfg.StochDPeriod);
                var kArr = new double[n]; var dArr = new double[n];
                for (int i = 0; i < n; i++)
                {
                    if (map.TryGetValue(times[i], out var st))
                    { kArr[i] = (double)st.K / 50.0 - 1.0; dArr[i] = (double)st.D / 50.0 - 1.0; }
                    else kArr[i] = dArr[i] = double.NaN;
                }
                indList.Add(kArr); indList.Add(dArr);
            }

            // ── MACD (single set) ─────────────────────────────────────────────
            if (cfg.MacdFast > 0 && cfg.MacdSlow > 0 && cfg.MacdSignal > 0)
            {
                var map = PriceIndicators.Macd(dict, cfg.MacdFast, cfg.MacdSlow, cfg.MacdSignal);
                var lineArr = new double[n]; var sigArr = new double[n]; var histArr = new double[n];
                for (int i = 0; i < n; i++)
                {
                    if (map.TryGetValue(times[i], out var md) && close[i] != 0)
                    { lineArr[i] = (double)md.Line / close[i]; sigArr[i] = (double)md.Signal / close[i]; histArr[i] = (double)md.Histogram / close[i]; }
                    else lineArr[i] = sigArr[i] = histArr[i] = double.NaN;
                }
                indList.Add(lineArr); indList.Add(sigArr); indList.Add(histArr);
            }

            // ── Candle shape (normalized by first ATR period) ──────────────────
            if (cfg.AtrPeriods.Length > 0)
            {
                var crMap = PriceIndicators.CandleRange(dict);
                var bsMap = PriceIndicators.BodySize(dict);
                var uwMap = PriceIndicators.UpperWick(dict);
                var lwMap = PriceIndicators.LowerWick(dict);
                var firstAtr = atrMaps[0];
                var crArr = new double[n]; var bsArr = new double[n];
                var uwArr = new double[n]; var lwArr = new double[n];
                for (int i = 0; i < n; i++)
                {
                    var raw = firstAtr.TryGetValue(times[i], out var av) ? (double)av : double.NaN;
                    crArr[i] = crMap.TryGetValue(times[i], out var cr) && !double.IsNaN(raw) && raw > 0 ? (double)cr * close[i] / raw : double.NaN;
                    bsArr[i] = bsMap.TryGetValue(times[i], out var bs) && !double.IsNaN(raw) && raw > 0 ? (double)bs * close[i] / raw : double.NaN;
                    uwArr[i] = uwMap.TryGetValue(times[i], out var uw) && !double.IsNaN(raw) && raw > 0 ? (double)uw * close[i] / raw : double.NaN;
                    lwArr[i] = lwMap.TryGetValue(times[i], out var lw) && !double.IsNaN(raw) && raw > 0 ? (double)lw * close[i] / raw : double.NaN;
                }
                indList.Add(crArr); indList.Add(bsArr); indList.Add(uwArr); indList.Add(lwArr);
            }

            // ── Distance from N-bar high / low ────────────────────────────────
            if (cfg.DistN > 0)
            {
                var dhMap = PriceIndicators.DistanceFromHighN(dict, cfg.DistN);
                var dlMap = PriceIndicators.DistanceFromLowN(dict, cfg.DistN);
                var dhArr = new double[n]; var dlArr = new double[n];
                for (int i = 0; i < n; i++)
                {
                    dhArr[i] = dhMap.TryGetValue(times[i], out var dh) ? (double)dh : double.NaN;
                    dlArr[i] = dlMap.TryGetValue(times[i], out var dl) ? (double)dl : double.NaN;
                }
                indList.Add(dhArr); indList.Add(dlArr);
            }

            // ── AccumDist z-score + AccumDistOsc ──────────────────────────────
            if (cfg.AdZScorePeriod > 0)
            {
                var adRaw    = BuildIndicatorLookup(symInds, $"ad/{tf}");
                var adoscRaw = BuildIndicatorLookup(symInds, $"adosc/{tf}");
                var adTemp   = new double[n];
                var adoscArr = new double[n];
                for (int i = 0; i < n; i++)
                {
                    adTemp[i]   = adRaw.TryGetValue(times[i], out var adv) ? adv : double.NaN;
                    adoscArr[i] = adoscRaw.TryGetValue(times[i], out var ao) && close[i] != 0 ? ao / close[i] : double.NaN;
                }
                indList.Add(RollingZScore(adTemp, cfg.AdZScorePeriod));
                indList.Add(adoscArr);
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[BuildSeriesData] Exception for symbol={symbol} tf={tf}: {ex.Message}");
            Console.Error.WriteLine($"  Candle count: {n}");
            if (n > 0)
            {
                Console.Error.WriteLine($"  First: {candles[0].Datetime:yyyy-MM-dd HH:mm:ss}  C={candles[0].Close}");
                Console.Error.WriteLine($"  Last:  {candles[n-1].Datetime:yyyy-MM-dd HH:mm:ss}  C={candles[n-1].Close}");
            }
            throw;
        }

        return new SeriesData { Times = times, Close = close, Indicators = indList.ToArray() };
    }

    private static Dictionary<DateTime, double> BuildIndicatorLookup(
        Dictionary<string, List<IndicatorValue>>? symInds, string key)
    {
        if (symInds == null || !symInds.TryGetValue(key, out var list))
            return new Dictionary<DateTime, double>(0);
        var d = new Dictionary<DateTime, double>(list.Count);
        foreach (var v in list)
            if (!v.IsFilled) d[v.Datetime] = (double)v.Value;
        return d;
    }

    private static double[] RollingZScore(double[] arr, int period)
    {
        var result = new double[arr.Length];
        for (int i = 0; i < arr.Length; i++)
        {
            if (i < period - 1 || double.IsNaN(arr[i])) { result[i] = double.NaN; continue; }
            double sum = 0, sumSq = 0; int cnt = 0;
            for (int j = i - period + 1; j <= i; j++)
            {
                if (double.IsNaN(arr[j])) continue;
                sum += arr[j]; sumSq += arr[j] * arr[j]; cnt++;
            }
            if (cnt < 2) { result[i] = double.NaN; continue; }
            double mean = sum / cnt;
            double std  = Math.Sqrt(Math.Max(0, sumSq / cnt - mean * mean));
            result[i] = std == 0 ? 0 : (arr[i] - mean) / std;
        }
        return result;
    }

    // ── Private: hot-path write helpers ──────────────────────────────────────

    private static void WriteShortAgg(
        double?[] vals, SeriesData sd, int lo, int hi,
        int[][] indStatIdx, int nInds, int nStats, string[] aggStats)
    {
        for (int ii = 0; ii < nInds; ii++)
        {
            var arr    = sd.Indicators[ii];
            var idxRow = indStatIdx[ii];  // int[] — stat index → column index

            double sum = 0, mn = double.MaxValue, mx = double.MinValue, last = double.NaN, sumSq = 0;
            int valid = 0;
            for (int k = lo; k <= hi; k++)
            {
                var v = arr[k];
                if (double.IsNaN(v)) continue;
                sum += v; sumSq += v * v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                last = v;
                valid++;
            }
            if (valid == 0) continue;

            double mean = sum / valid;
            double std  = valid > 1 ? Math.Sqrt(Math.Max(0, sumSq / valid - mean * mean)) : 0.0;

            for (int si = 0; si < nStats; si++)
            {
                int ci = idxRow[si];
                if (ci < 0) continue;
                vals[ci] = aggStats[si] switch
                {
                    "mean" => mean,
                    "min"  => mn == double.MaxValue ? null : (double?)mn,
                    "max"  => mx == double.MinValue ? null : (double?)mx,
                    "last" => double.IsNaN(last) ? null : (double?)last,
                    "std"  => std,
                    _      => null
                };
            }
        }
    }

    private static void WriteLong(double?[] vals, SeriesData sd, int i, int[] indIdx, int nInds)
    {
        for (int ii = 0; ii < nInds; ii++)
        {
            int ci = indIdx[ii];
            if (ci < 0) continue;
            var v = sd.Indicators[ii][i];
            if (!double.IsNaN(v)) vals[ci] = v;
        }
    }

    private static void WriteCrossAsset(
        double?[] vals, DateTime T,
        SeriesData?[][] longSd, SeriesData?[][] shortSd,
        double goldSilverZ, double goldOilZ,
        int tltAi, int shyAi,
        int spyAi, int vixyAi, int udnAi,
        int tf4hLongTi, int tf1hShortTi,
        int[] xaCols,
        int indLogReturn, int indRsi, int indAtr)
    {
        static double FloorClose(SeriesData?[][] ld, int ai, int ti, DateTime t)
        {
            if (ai < 0 || ti < 0) return double.NaN;
            var sd = ld[ai][ti];
            if (sd == null) return double.NaN;
            int i = BsFloor(sd.Times, t);
            return i >= 0 ? sd.Close[i] : double.NaN;
        }
        static double FloorInd(SeriesData?[][] ld, int ai, int ti, DateTime t, int indIdx)
        {
            if (ai < 0 || ti < 0) return double.NaN;
            var sd = ld[ai][ti];
            if (sd == null) return double.NaN;
            int i = BsFloor(sd.Times, t);
            return i >= 0 ? sd.Indicators[indIdx][i] : double.NaN;
        }

        void Set(int xaIdx, double v) { if (xaCols[xaIdx] >= 0 && !double.IsNaN(v)) vals[xaCols[xaIdx]] = v; }

        double tlt    = FloorClose(longSd, tltAi,  tf4hLongTi, T);
        double shy    = FloorClose(longSd, shyAi,  tf4hLongTi, T);

        Set(0, goldSilverZ);     // GoldSilverRatio (z-scored)
        Set(1, goldOilZ);        // GoldOilRatio    (z-scored)

        // DXY mom via UDN LogReturn — 1h short TF and 4h long TF
        Set(2, FloorInd(shortSd, udnAi, tf1hShortTi, T, indLogReturn)); // DXY_Mom_1h
        Set(3, FloorInd(longSd,  udnAi, tf4hLongTi,  T, indLogReturn)); // DXY_Mom_4h

        Set(4, shy > 0 ? tlt / shy : double.NaN);            // YieldCurveProxy

        // RiskSentiment: SPY RSI - VIXY RSI
        double spyRsi  = FloorInd(longSd, spyAi,  tf4hLongTi, T, indRsi);
        double vixyRsi = FloorInd(longSd, vixyAi, tf4hLongTi, T, indRsi);
        Set(5, !double.IsNaN(spyRsi) && !double.IsNaN(vixyRsi) ? spyRsi - vixyRsi : double.NaN);

        // VolatilityRegime: VIXY ATR/close bucketed into 0/1/2
        double vixyAtr = FloorInd(longSd, vixyAi, tf4hLongTi, T, indAtr);
        if (!double.IsNaN(vixyAtr))
            Set(6, vixyAtr < 0.02 ? 0.0 : vixyAtr < 0.05 ? 1.0 : 2.0);
    }

    // ── Private: indicator name generation ───────────────────────────────────

    private static string[] BuildIndicatorNames(DatasetConfig cfg)
    {
        var names = new List<string>();
        names.Add("LogReturn");
        foreach (var p in cfg.SmaPeriods)       names.Add($"SMA_{p}");
        foreach (var p in cfg.EmaPeriods)       names.Add($"EMA_{p}");
        for (int k = 0; k + 1 < cfg.SmaPeriods.Length; k++) names.Add($"SMASpread_{cfg.SmaPeriods[k]}_{cfg.SmaPeriods[k + 1]}");
        for (int k = 0; k + 1 < cfg.EmaPeriods.Length; k++) names.Add($"EMASpread_{cfg.EmaPeriods[k]}_{cfg.EmaPeriods[k + 1]}");
        foreach (var p in cfg.RsiPeriods)       names.Add($"RSI_{p}");
        foreach (var p in cfg.RocPeriods)       names.Add($"ROC_{p}");
        foreach (var p in cfg.StdDevPeriods)    names.Add($"StdDev_{p}");
        foreach (var p in cfg.AtrPeriods)       names.Add($"ATR_{p}");
        foreach (var p in cfg.AdxPeriods)       { names.Add($"ADX_{p}"); names.Add($"PlusDI_{p}"); names.Add($"MinusDI_{p}"); }
        foreach (var p in cfg.BbPeriods)
        {
            names.Add($"BbUpper_{p}"); names.Add($"BbLower_{p}"); names.Add($"BbWidth_{p}");
            if (cfg.BbWidthZPeriod > 0) names.Add($"BbWidthZ_{p}");
        }
        foreach (var p in cfg.CciPeriods)       names.Add($"CCI_{p}");
        foreach (var p in cfg.WilliamsRPeriods) names.Add($"WilliamsR_{p}");
        foreach (var p in cfg.StochKPeriods)    { names.Add($"StochK_{p}"); names.Add($"StochD_{p}"); }
        if (cfg.MacdFast > 0 && cfg.MacdSlow > 0 && cfg.MacdSignal > 0)
            { names.Add("MacdLine"); names.Add("MacdSignal"); names.Add("MacdHist"); }
        if (cfg.AtrPeriods.Length > 0)
            { names.Add("CandleRange"); names.Add("BodySize"); names.Add("UpperWick"); names.Add("LowerWick"); }
        if (cfg.DistN > 0)
            { names.Add("DistHigh"); names.Add("DistLow"); }
        if (cfg.AdZScorePeriod > 0)
            { names.Add("AccumDist"); names.Add("AccumDistOsc"); }
        return names.ToArray();
    }

    // ── Private: utilities ────────────────────────────────────────────────────

    /// <summary>Largest index where Times[i] &lt;= target, or -1 if none.</summary>
    private static int BsFloor(DateTime[] arr, DateTime target)
    {
        int lo = 0, hi = arr.Length - 1, res = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (arr[mid] <= target) { res = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return res;
    }

    private static string SafeSymbol(string s) =>
        s.Replace("/", "").Replace("_", "").Replace("-", "").ToUpperInvariant();

    private static int GetSession(DateTime utc) =>
        utc.Hour < 8  ? 0 :   // Asia
        utc.Hour < 13 ? 1 :   // London
        2;                     // NY

    private static double Percentile(double[] sorted, double p)
    {
        if (sorted.Length == 0) return 0;
        double pos = p / 100.0 * (sorted.Length - 1);
        int lo = (int)pos, hi = Math.Min(lo + 1, sorted.Length - 1);
        return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
    }

    private static int GetQuintile(double value, double[] boundaries)
    {
        for (int i = 0; i < boundaries.Length; i++)
            if (value <= boundaries[i]) return i;
        return boundaries.Length;
    }
}
