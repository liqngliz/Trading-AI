using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Imputers;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Predictor;

public static class ModelTrainer
{
    private static readonly (string Suffix, string Label, string Imputer)[] TrainBuckets =
    [
        ("nan_0_66",   "≤66% NaN",  "knn"),
        ("nan_66_100", ">66% NaN",  "mean"),
    ];

    private const double Pass2FeatureFraction = 0.40;
    private const int    MinRows              = 300;

    // ── Black swan exclusion windows ──────────────────────────────────────────
    // Each entry covers: event_start − 14 days  →  event_end + 7 days.
    // Assets = null  →  excluded for every target (universal macro shock).
    // Assets = [...] →  excluded only when the target column starts with one
    //                   of the listed safe-symbol prefixes (e.g. "XAUUSD").
    private static readonly (DateTime From, DateTime To, string Name, string[]? Assets)[] BlackSwanPeriods =
    [
        // ── Universal macro shocks ─────────────────────────────────────────────
        (new(1929, 10, 15), new(1932,  8,  8), "Wall Street Crash / Great Depression", null),
        (new(1973,  9, 17), new(1974,  4,  8), "OPEC Oil Embargo",                     null),
        (new(1987, 10,  5), new(1987, 10, 26), "Black Monday",                         null),
        (new(1997,  6, 17), new(1999,  1,  7), "Asian Financial Crisis",               null),
        (new(1998,  8,  3), new(1998,  8, 24), "Russian Default / LTCM",               null),
        (new(2000,  2, 15), new(2002, 10,  8), "Dot-com Bubble Burst",                 null),
        (new(2001,  8, 28), new(2001,  9, 18), "9/11 Attacks",                         null),
        (new(2008,  9,  1), new(2009,  3,  8), "Lehman / Global Financial Crisis",     null),
        (new(2010,  4, 22), new(2010,  5, 13), "Flash Crash",                          null),
        (new(2015,  8, 10), new(2015,  8, 31), "China Yuan Devaluation",               null),
        (new(2016,  6,  9), new(2016,  6, 30), "Brexit Vote",                          null),
        (new(2020,  2,  6), new(2020,  3, 30), "COVID-19 Pandemic",                    null),

        // ── Gold & silver specific ─────────────────────────────────────────────
        // Gulf War I: Iraq invaded Kuwait Aug 2 1990; oil/gold spiked sharply
        (new(1990,  7, 19), new(1991,  3,  7), "Gulf War I",
            ["XAUUSD", "XAGUSD", "WTIUSD"]),
        // Brown's Bottom: UK announced May 7 1999 sale of 395t gold → "Brown's Bottom"
        // Washington Agreement Sep 26 1999 stabilised but window covers both
        (new(1999,  4, 23), new(1999, 10,  3), "Brown's Bottom / Washington Gold Agreement",
            ["XAUUSD", "XAGUSD"]),
        // Gold Flash Crash: Goldman sell note Apr 10 2013; gold fell $200 in 2 days
        (new(2013,  3, 29), new(2013,  4, 22), "Gold Flash Crash",
            ["XAUUSD", "XAGUSD"]),
        // SNB CHF/EUR peg removal Jan 15 2015: gold and CHF spiked violently
        (new(2015,  1,  1), new(2015,  1, 22), "SNB CHF Peg Removal",
            ["XAUUSD", "XAGUSD"]),
        // Ukraine invasion Feb 24 2022: gold/silver spiked; oil to 14-year highs
        (new(2022,  2, 10), new(2022,  3, 25), "Russia-Ukraine War Outbreak",
            ["XAUUSD", "XAGUSD", "WTIUSD"]),

        // ── 2026 Geopolitical shocks ──────────────────────────────────────────
        // Liberation Day Apr 2 2026: sweeping US tariffs announced; global market rout
        (new(2026,  3, 19), new(2026,  4,  9), "Liberation Day Tariff Shock",           null),
        // US-Iran War / Strait of Hormuz closure Mar 12 2026: ~20% of global oil supply
        // disrupted; energy, gold, equities in extreme dislocation; ongoing
        (new(2026,  2, 26), new(2026,  7, 31), "US-Iran War / Strait of Hormuz Closure", null),

        // ── Oil specific ───────────────────────────────────────────────────────
        // Hurricane Katrina Aug 29 2005: 25% of US Gulf oil production offline
        (new(2005,  8, 15), new(2005,  9, 19), "Hurricane Katrina",
            ["WTIUSD", "SPY", "QQQ"]),
        // Arab Spring / Libya Feb-Mar 2011: Libya's 1.6 mb/d output collapsed
        (new(2011,  2,  3), new(2011,  4, 30), "Arab Spring / Libya Oil Disruption",
            ["WTIUSD"]),
        // OPEC Nov 2014: Saudi Arabia refused to cut; oil crashed $100→$26 over 15 months
        (new(2014, 11, 13), new(2016,  3,  7), "OPEC Price War / Oil Collapse",
            ["WTIUSD", "SPY", "QQQ"]),
        // Saudi-Russia price war + negative WTI Apr 20 2020 (after COVID window ends)
        (new(2020,  4,  6), new(2020,  5,  4), "WTI Negative Price",
            ["WTIUSD"]),
    ];

    // targetPrefix: the safe-symbol prefix of the target column, e.g. "XAUUSD"
    private static bool IsNearBlackSwan(DateTime ts, string targetPrefix) =>
        BlackSwanPeriods.Any(p =>
            ts >= p.From && ts <= p.To &&
            (p.Assets is null || p.Assets.Any(a => targetPrefix.StartsWith(a, StringComparison.Ordinal))));

    public static void Run(string datasetDir, string modelDir, int numFolds = 10, double holdoutPct = 0.15,
                           int purge = 1, int embargo = 50, int? holdoutDays = null)
    {
        var mlContext = new MLContext(seed: 42);

        var baseCsv  = Directory.GetFiles(datasetDir, "*_dataset.csv").FirstOrDefault();
        var baseName = baseCsv is not null
            ? Path.GetFileNameWithoutExtension(baseCsv)
            : Path.GetFileName(datasetDir) + "_dataset";

        var bucketCsvs = TrainBuckets
            .Select(b => (
                b.Label,
                b.Suffix,
                b.Imputer,
                Path: Path.Combine(datasetDir, $"{baseName}_{b.Suffix}.csv")))
            .ToArray();

        var missing = bucketCsvs.Where(b => !File.Exists(b.Path)).ToArray();
        if (missing.Length == bucketCsvs.Length)
        {
            Console.WriteLine("ERROR: No NaN-bucket CSVs found. Run the Importer first.");
            Console.WriteLine($"  Expected files like: {bucketCsvs[0].Path}");
            return;
        }
        foreach (var m in missing)
            Console.WriteLine($"[WARN] Bucket CSV not found, skipping: {m.Path}");

        var firstCsv  = bucketCsvs.First(b => File.Exists(b.Path)).Path;
        var allCols    = File.ReadLines(firstCsv).First().Split(',');
        var targetCols = allCols
            .Where(c => c.Contains("_Target_") && c.EndsWith("_Return"))
            .ToArray();

        if (targetCols.Length == 0)
        {
            Console.WriteLine("ERROR: No target columns found in dataset.");
            return;
        }

        Console.WriteLine($"Targets  : {string.Join(", ", targetCols)}");
        Console.WriteLine($"Buckets  : {bucketCsvs.Length}");
        Console.WriteLine($"CV folds : {numFolds}");

        foreach (var targetCol in targetCols)
        {
            Console.WriteLine($"\n{new string('=', 70)}");
            Console.WriteLine($"Target: {targetCol}");
            Console.WriteLine(new string('=', 70));

            foreach (var (label, suffix, imputer, csvPath) in bucketCsvs)
            {
                if (!File.Exists(csvPath))
                    continue;

                Console.WriteLine($"\n── Bucket: {label} ({suffix}) ──────────────────────────");

                var cols        = File.ReadLines(csvPath).First().Split(',');
                var featureCols = cols.Skip(1).Where(c =>
                    !c.Contains("_Target_") &&
                    !c.EndsWith("_BarClose", StringComparison.Ordinal) &&
                    !c.EndsWith("_TargetVolScalar", StringComparison.Ordinal)).ToArray();
                var colIndex    = cols.Select((n, i) => (n, i)).ToDictionary(x => x.n, x => x.i);

                if (!colIndex.TryGetValue(targetCol, out var targetIdx))
                {
                    Console.WriteLine($"  [SKIP] Target column not found in {suffix} CSV.");
                    continue;
                }

                var featureIdx = featureCols.Select(c => colIndex[c]).ToArray();
                Console.WriteLine($"  Features : {featureCols.Length}");

                var sw = Stopwatch.StartNew();

                var (rawF, labels, timestamps) = LoadCsv(csvPath, featureIdx, targetIdx);
                if (labels.Length == 0)
                {
                    Console.WriteLine($"  [SKIP] No labelled rows in {suffix} CSV.");
                    continue;
                }
                if (labels.Length < MinRows)
                {
                    Console.WriteLine($"  [SKIP] Too few rows ({labels.Length}) in {suffix} CSV — minimum is {MinRows}.");
                    continue;
                }

                // ── Hold-out split: last N days or last N% by time ───────────
                int hoSize;
                if (holdoutDays.HasValue && timestamps.Length > 0)
                {
                    var cutoff = timestamps.Last().AddDays(-holdoutDays.Value);
                    hoSize = timestamps.Count(t => t >= cutoff);
                }
                else
                {
                    hoSize = (int)(labels.Length * holdoutPct);
                }
                int    cvSize     = labels.Length - hoSize;
                var    cvRawF     = hoSize > 0 ? rawF[..cvSize]        : rawF;
                var    cvLabels   = hoSize > 0 ? labels[..cvSize]      : labels;
                var    hoRawF     = hoSize > 0 ? rawF[cvSize..]        : [];
                var    hoLabels   = hoSize > 0 ? labels[cvSize..]      : [];
                var    hoTs       = hoSize > 0 ? timestamps[cvSize..]  : [];

                // ── Clean hold-out: exclude rows near black swan events ────────
                var targetPrefix = targetCol.Contains('_') ? targetCol[..targetCol.IndexOf('_')] : targetCol;
                var cleanIdx    = Enumerable.Range(0, hoLabels.Length)
                    .Where(i => hoTs.Length == 0 || !IsNearBlackSwan(hoTs[i], targetPrefix))
                    .ToArray();
                var cleanHoRawF  = cleanIdx.Select(i => hoRawF[i]).ToArray();
                var cleanHoLabels = cleanIdx.Select(i => hoLabels[i]).ToArray();
                int blackSwanCount = hoLabels.Length - cleanIdx.Length;

                Console.WriteLine($"  Rows     : {labels.Length}  (CV={cvLabels.Length}  hold-out={hoLabels.Length}  clean-hold-out={cleanHoLabels.Length}  black-swan-excluded={blackSwanCount})");

                var summary = new List<string>();
                void S(string line) { Console.WriteLine(line); summary.Add(line); }

                // Summary header
                S($"Target  : {targetCol}");
                S($"Bucket  : {label} ({suffix})");
                S($"Rows    : {labels.Length}  (CV={cvLabels.Length}  hold-out={hoLabels.Length})");
                S($"Imputer : {imputer}");
                S($"Folds   : {numFolds}  (walk-forward expanding window)");
                S($"Purge   : {purge} bar(s)  |  Embargo: {embargo} bar(s)");
                var holdoutDesc = holdoutDays.HasValue
                    ? $"last {holdoutDays.Value} days ({hoSize} rows)"
                    : $"last {holdoutPct:P0} by time  ({hoSize} rows)";
                S($"Holdout : {holdoutDesc}");
                S(new string('─', 70));

                // ── Helper: slice raw feature arrays to a column subset ───────
                float[][] SliceCols(float[][] data, int[] idx) =>
                    data.Select(row => idx.Select(i => row[i]).ToArray()).ToArray();

                int[] ColIndices(HashSet<int> keep) =>
                    featureCols.Select((_, i) => i).Where(keep.Contains).ToArray();

                // ── Pass 1: all features ──────────────────────────────────────
                S($"\npass 1 ({featureCols.Length} features):");
                var cvLines1 = RunCV(mlContext, imputer, cvRawF, cvLabels, featureCols, numFolds, purge, embargo);
                foreach (var l in cvLines1) S(l);

                var (model1, hoRmse1, cvF1, hoF1) = TrainFinalModel(
                    mlContext, imputer, cvRawF, cvLabels, hoRawF, hoLabels, featureCols);
                var (p1Ho, _) = FormatHoldOut(hoLabels, mlContext, model1, hoF1, featureCols.Length, sw);
                S($"  Hold-out (pass 1): {p1Ho.Trim()}");

                // ── Derive pass 2 feature subset from pass 1 gain ranking ────
                var gainRanking = GetGainRanking(model1, featureCols);

                int keepN2       = Math.Max(1, (int)Math.Ceiling(featureCols.Length * Pass2FeatureFraction));
                var topIdx2      = gainRanking.Take(keepN2).Select(x => x.Idx).ToHashSet();
                var featureCols2 = featureCols.Where((_, i) => topIdx2.Contains(i)).ToArray();
                var localIdx2    = ColIndices(topIdx2);
                var cvRawF2      = SliceCols(cvRawF, localIdx2);
                var hoRawF2      = SliceCols(hoRawF, localIdx2);

                // ── Pass 2: top 40% ───────────────────────────────────────────
                S($"\npass 2 (top {keepN2}/{featureCols.Length} = 40% features):");
                var cvLines2 = RunCV(mlContext, imputer, cvRawF2, cvLabels, featureCols2, numFolds, purge, embargo);
                foreach (var l in cvLines2) S(l);

                var (model2, hoRmse2, cvF2, hoF2) = TrainFinalModel(
                    mlContext, imputer, cvRawF2, cvLabels, hoRawF2, hoLabels, featureCols2);
                var (p2Ho, _) = FormatHoldOut(hoLabels, mlContext, model2, hoF2, featureCols2.Length, sw);
                S($"  Hold-out (pass 2): {p2Ho.Trim()}");

                // ── Pick winner ───────────────────────────────────────────────
                double bestRmse = Math.Min(hoRmse1, hoRmse2);
                var winModel = bestRmse == hoRmse2 ? model2       : model1;
                var winCols  = bestRmse == hoRmse2 ? featureCols2 : featureCols;
                var winCvF   = bestRmse == hoRmse2 ? cvF2         : cvF1;
                var winPass  = bestRmse == hoRmse2 ? "pass 2"     : "pass 1";

                S($"\n{new string('─', 70)}");
                S($"  {"Pass",-8} {"Features",10}  {"Hold-out RMSE",15}  {"Winner",8}");
                S($"  {new string('─', 50)}");
                S($"  {"pass 1",-8} {featureCols.Length,10}  {hoRmse1,15:F6}  {(winPass == "pass 1" ? "★" : "")}");
                S($"  {"pass 2",-8} {keepN2,10}  {hoRmse2,15:F6}  {(winPass == "pass 2" ? "★" : "")}");
                S($"  Winner : {winPass}  ({winCols.Length} features)");

                // ── Clean hold-out evaluation (winner model) ──────────────────
                var winHoLocalIdx = bestRmse == hoRmse2 ? localIdx2
                                  : Enumerable.Range(0, featureCols.Length).ToArray();
                // Imputation reference: always the winning CV training set, never holdout data.
                var winCvSliced = SliceCols(cvRawF, winHoLocalIdx);

                S($"\n{new string('─', 70)}");
                S($"  Hold-out evaluation  (winner: {winPass})");
                S($"  {"Set",-35} {"RMSE",10}  {"MAE",10}  {"R²",7}");
                S($"  {new string('─', 65)}");

                // Random hold-out
                string HoMetrics(float[] hLabels, float[][] hF) {
                    if (hLabels.Length < 10) return $"  (too few rows: {hLabels.Length})";
                    var hDv = ToDataView(mlContext, hF, hLabels, winCols.Length);
                    var m   = mlContext.Regression.Evaluate(winModel.Transform(hDv), labelColumnName: "Label");
                    return $"  {$"Random hold-out ({hLabels.Length} rows)",-35} {m.RootMeanSquaredError,10:F6}  {m.MeanAbsoluteError,10:F6}  {m.RSquared,7:F4}";
                }
                var (_, winHoF) = Impute(imputer, winCvSliced, SliceCols(hoRawF, winHoLocalIdx));
                S(HoMetrics(hoLabels, winHoF));

                // Clean hold-out
                var cleanHoSliced = cleanHoRawF.Length > 0
                    ? cleanHoRawF.Select(row => winHoLocalIdx.Select(i => row[i]).ToArray()).ToArray()
                    : [];
                double cleanHoldoutRmse = double.NaN;
                if (cleanHoLabels.Length >= 10)
                {
                    var (_, cleanHoF) = Impute(imputer, winCvSliced, cleanHoSliced);
                    var cleanDv  = ToDataView(mlContext, cleanHoF, cleanHoLabels, winCols.Length);
                    var cleanM   = mlContext.Regression.Evaluate(winModel.Transform(cleanDv), labelColumnName: "Label");
                    S($"  {$"Clean hold-out (excl. {blackSwanCount} BS rows)",-35} {cleanM.RootMeanSquaredError,10:F6}  {cleanM.MeanAbsoluteError,10:F6}  {cleanM.RSquared,7:F4}");

                    cleanHoldoutRmse = cleanM.RootMeanSquaredError;
                    if (!double.IsNaN(cleanHoldoutRmse) && cleanHoldoutRmse > 0)
                    {
                        double ratio = bestRmse / cleanHoldoutRmse;
                        string tag = ratio > 2.0 ? "*** STRONG WARNING" : ratio > 1.5 ? "** WARNING" : ratio > 1.2 ? "* MILD" : "OK";
                        S($"  Full/clean RMSE ratio = {ratio:F3}  [{tag}]");
                    }
                }

                // Temporal hold-out
                var (tempAllF, tempAllL, tempAllTs) = LoadCsv(csvPath, featureIdx, targetIdx);
                int tempHoStart  = Math.Max(0, tempAllL.Length - hoSize);
                var tempHoLabels = tempAllL[tempHoStart..];
                var (_, tempHoF) = Impute(imputer, winCvSliced, SliceCols(tempAllF[tempHoStart..], winHoLocalIdx));
                if (tempHoLabels.Length >= 10)
                {
                    var tempDv = ToDataView(mlContext, tempHoF, tempHoLabels, winCols.Length);
                    var tempM  = mlContext.Regression.Evaluate(winModel.Transform(tempDv), labelColumnName: "Label");
                    S($"  {$"Temporal hold-out (last {holdoutPct:P0}, {tempHoLabels.Length} rows)",-35} {tempM.RootMeanSquaredError,10:F6}  {tempM.MeanAbsoluteError,10:F6}  {tempM.RSquared,7:F4}");
                }
                else
                {
                    S($"  {"Temporal hold-out",-35} (too few rows: {tempHoLabels.Length})");
                }

                // Black-swan hold-out: rows that are BOTH in the holdout window AND near a BS event
                var bsIdx      = Enumerable.Range(tempHoStart, tempAllL.Length - tempHoStart)
                    .Where(i => tempAllTs.Length > i && IsNearBlackSwan(tempAllTs[i], targetPrefix))
                    .ToArray();
                var bsHoLabels = bsIdx.Select(i => tempAllL[i]).ToArray();
                var bsHoSliced = SliceCols([.. bsIdx.Select(i => tempAllF[i])], winHoLocalIdx);
                if (bsHoLabels.Length >= 10)
                {
                    var (_, bsHoF) = Impute(imputer, winCvSliced, bsHoSliced);
                    var bsDv = ToDataView(mlContext, bsHoF, bsHoLabels, winCols.Length);
                    var bsM  = mlContext.Regression.Evaluate(winModel.Transform(bsDv), labelColumnName: "Label");
                    S($"  {$"Black-swan hold-out ({bsHoLabels.Length} rows)",-35} {bsM.RootMeanSquaredError,10:F6}  {bsM.MeanAbsoluteError,10:F6}  {bsM.RSquared,7:F4}");
                }
                else
                {
                    S($"  {"Black-swan hold-out",-35} (too few rows: {bsHoLabels.Length})");
                }

                // Holdout chunks (5 equal temporal segments)
                var tempHoAllF  = tempAllF[tempHoStart..];
                var tempHoAllTs = tempAllTs.Length > tempHoStart ? tempAllTs[tempHoStart..] : [];
                if (tempHoLabels.Length >= 10)
                {
                    const int NumChunks = 5;
                    int chunkLen = tempHoLabels.Length / NumChunks;
                    S($"\n  Holdout chunks ({NumChunks} × ~{chunkLen} rows):");
                    S($"  {"Chunk",-6} {"From",-12} {"To",-12} {"Rows",5}   {"RMSE",10}  {"MAE",10}  {"R²",7}");
                    S($"  {new string('─', 70)}");
                    for (int c = 0; c < NumChunks; c++)
                    {
                        int cStart = c * chunkLen;
                        int cEnd   = c == NumChunks - 1 ? tempHoLabels.Length : (c + 1) * chunkLen;
                        var cLabels = tempHoLabels[cStart..cEnd];
                        var cF      = SliceCols(tempHoAllF[cStart..cEnd], winHoLocalIdx);
                        var (_, cImpF) = Impute(imputer, winCvSliced, cF);
                        var cDv = ToDataView(mlContext, cImpF, cLabels, winCols.Length);
                        var cM  = mlContext.Regression.Evaluate(winModel.Transform(cDv), labelColumnName: "Label");
                        string fromStr = tempHoAllTs.Length > cStart ? tempHoAllTs[cStart].ToString("yyyy-MM-dd") : "?";
                        string toStr   = tempHoAllTs.Length > cEnd - 1 ? tempHoAllTs[cEnd - 1].ToString("yyyy-MM-dd") : "?";
                        S($"  {c + 1,-6} {fromStr,-12} {toStr,-12} {cLabels.Length,5}   {cM.RootMeanSquaredError,10:F6}  {cM.MeanAbsoluteError,10:F6}  {cM.RSquared,7:F4}");
                    }
                }

                sw.Stop();
                S($"\n  Total training time: {sw.Elapsed.TotalSeconds:F1}s");

                // ── Top 30 features of winner model ───────────────────────────
                summary.Add($"\nTop 30 features ({winPass}):");
                summary.Add($"  {"#",-4} {"Feature",-55} {"Gain",10}  {"Corr",7}");
                summary.Add($"  {new string('-', 82)}");
                var top30 = GetGainRanking(winModel, winCols).Take(30).ToArray();
                for (int i = 0; i < top30.Length; i++)
                {
                    string fname   = top30[i].Idx < winCols.Length ? winCols[top30[i].Idx] : $"f{top30[i].Idx}";
                    double corr    = PearsonCorr(winCvF, cvLabels, top30[i].Idx);
                    string corrStr = double.IsNaN(corr) ? "    n/a" : $"{corr:+0.000;-0.000}";
                    summary.Add($"  {i + 1,-4} {fname,-55} {top30[i].Score,10:F4}  {corrStr}");
                }

                // ── Save winner model ─────────────────────────────────────────
                var modelName = $"{targetCol}_{suffix}";
                var modelPath = Path.Combine(modelDir, $"{modelName}.zip");
                var prodDv    = ToDataView(mlContext, winCvF, cvLabels, winCols.Length);
                mlContext.Model.Save(winModel, prodDv.Schema, modelPath);
                Console.WriteLine($"  Saved → {modelPath}");

                var (featMean, featStd) = ComputeFeatureStats(winCvF);

                // Build gain array aligned to winCols order (0.0 for features not in ranking)
                var gainRankFinal = GetGainRanking(winModel, winCols);
                var gainByIdx     = gainRankFinal.ToDictionary(x => x.Idx, x => (double)x.Score);
                var featGain      = Enumerable.Range(0, winCols.Length)
                                              .Select(i => gainByIdx.GetValueOrDefault(i, 0.0))
                                              .ToArray();

                var metaPath = Path.ChangeExtension(modelPath, ".features.json");
                File.WriteAllText(metaPath, JsonSerializer.Serialize(
                    new ModelMeta
                    {
                        TargetColumn   = targetCol,
                        NanBucket      = suffix,
                        Imputer        = imputer,
                        FeatureColumns = winCols,
                        FeatureMean    = featMean,
                        FeatureStd     = featStd,
                        FeatureGain    = featGain,
                    },
                    new JsonSerializerOptions { WriteIndented = true }));
                Console.WriteLine($"  Saved → {metaPath}");

                var rankPath = Path.ChangeExtension(modelPath, ".feature_ranking.txt");
                WriteFeatureRanking(winModel, winCols, winCvF, cvLabels, rankPath);
                Console.WriteLine($"  Saved → {rankPath}");

                var summaryPath = Path.ChangeExtension(modelPath, ".training_summary.txt");
                File.WriteAllLines(summaryPath, summary);
                Console.WriteLine($"  Saved → {summaryPath}");

                // Release large arrays before next bucket/target iteration
                rawF       = null!;
                cvRawF     = null!;
                hoRawF     = null!;
                cvRawF2    = null!;
                hoRawF2    = null!;
                winCvF     = null!;
                winCvSliced = null!;
                model1     = null!;
                model2     = null!;
                winModel   = null!;
                GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
            }
        }

        Console.WriteLine($"\n{new string('=', 70)}");
        Console.WriteLine("Training complete.");
    }

    // ── CV runner — returns fold table lines for summary ──────────────────────

    private static List<string> RunCV(
        MLContext  mlContext,
        string     imputer,
        float[][]  cvRawF,
        float[]    cvLabels,
        string[]   featureCols,
        int        numFolds,
        int        purge   = 1,
        int        embargo = 50)
    {
        var lines    = new List<string>();
        var cvRmse   = new List<double>();
        var cvMae    = new List<double>();
        var cvR2     = new List<double>();
        var foldRows = new List<(int FoldNum, int TrainLen, int ValLen)>();
        int foldNum  = 0;

        foreach (var (rawTrainF, trainL, rawValF, valL) in WalkForwardSplit(cvRawF, cvLabels, numFolds, purge, embargo))
        {
            foldNum++;
            Console.WriteLine($"\n  ── Fold {foldNum}/{numFolds}  train={trainL.Length}  val={valL.Length} ──");

            var (fTrainF, fValF) = Impute(imputer, rawTrainF, rawValF);
            var trainDv   = ToDataView(mlContext, fTrainF, trainL, featureCols.Length);
            var foldModel = BuildPipeline(mlContext).Fit(trainDv);

            PrintTopFeatures(foldModel, featureCols, fTrainF, trainL, topN: 30);

            var valDv = ToDataView(mlContext, fValF, valL, featureCols.Length);
            var fm    = mlContext.Regression.Evaluate(foldModel.Transform(valDv), labelColumnName: "Label");

            Console.WriteLine($"\n  RMSE={fm.RootMeanSquaredError:F6}  MAE={fm.MeanAbsoluteError:F6}  R²={fm.RSquared:F4}");

            cvRmse.Add(fm.RootMeanSquaredError);
            cvMae.Add(fm.MeanAbsoluteError);
            cvR2.Add(fm.RSquared);
            foldRows.Add((foldNum, trainL.Length, valL.Length));
        }

        lines.Add($"  {"Fold",-6} {"Train",6} {"Val",5}  {"RMSE",10}  {"MAE",10}  {"R²",7}");
        lines.Add($"  {new string('-', 55)}");
        for (int fi = 0; fi < foldRows.Count; fi++)
        {
            var (fn, tLen, vLen) = foldRows[fi];
            lines.Add($"  {fn,-6} {tLen,6} {vLen,5}  {cvRmse[fi],10:F6}  {cvMae[fi],10:F6}  {cvR2[fi],7:F4}");
        }
        if (cvRmse.Count > 0)
        {
            double mRmse  = cvRmse.Average();
            double sdRmse = Math.Sqrt(cvRmse.Average(v => (v - mRmse) * (v - mRmse)));
            lines.Add($"  {"mean",-6} {"",6} {"",5}  {mRmse,10:F6}  {cvMae.Average(),10:F6}  {cvR2.Average(),7:F4}  ±{sdRmse:F6}");
        }

        return lines;
    }

    // ── Final model training ──────────────────────────────────────────────────

    private static (ITransformer Model, double HoRmse, float[][] CvFeatures, float[][] HoFeatures)
        TrainFinalModel(
            MLContext  mlContext,
            string     imputer,
            float[][]  cvRawF,
            float[]    cvLabels,
            float[][]  hoRawF,
            float[]    hoLabels,
            string[]   featureCols)
    {
        var (cvF, hoF) = Impute(imputer, cvRawF, hoRawF);
        var dv    = ToDataView(mlContext, cvF, cvLabels, featureCols.Length);
        var model = BuildPipeline(mlContext).Fit(dv);

        double hoRmse = double.MaxValue;
        if (hoF.Length > 0)
        {
            var hoDv    = ToDataView(mlContext, hoF, hoLabels, featureCols.Length);
            var metrics = mlContext.Regression.Evaluate(model.Transform(hoDv), labelColumnName: "Label");
            hoRmse = metrics.RootMeanSquaredError;
        }

        return (model, hoRmse, cvF, hoF);
    }

    private static (string Line, double Rmse) FormatHoldOut(
        float[]      hoLabels,
        MLContext    mlContext,
        ITransformer model,
        float[][]    hoF,
        int          numFeatures,
        Stopwatch    sw) // sw kept for signature compat but no longer printed per-line
    {
        if (hoLabels.Length > 0 && hoF.Length > 0)
        {
            var hoDv    = ToDataView(mlContext, hoF, hoLabels, numFeatures);
            var metrics = mlContext.Regression.Evaluate(model.Transform(hoDv), labelColumnName: "Label");
            return (
                $"RMSE={metrics.RootMeanSquaredError:F6}  MAE={metrics.MeanAbsoluteError:F6}  R²={metrics.RSquared:F4}",
                metrics.RootMeanSquaredError);
        }
        return ($"  (no hold-out rows)  ({sw.Elapsed.TotalSeconds:F1}s)", double.NaN);
    }

    // ── Feature distribution stats (for drift detection) ─────────────────────

    private static (double[] Mean, double[] Std) ComputeFeatureStats(float[][] data)
    {
        if (data.Length == 0) return ([], []);
        int n = data[0].Length;
        var mean = new double[n];
        var std  = new double[n];
        for (int f = 0; f < n; f++)
        {
            double sum = 0; int count = 0;
            foreach (var row in data) if (!float.IsNaN(row[f])) { sum += row[f]; count++; }
            mean[f] = count > 0 ? sum / count : 0;
            double sum2 = 0;
            foreach (var row in data) if (!float.IsNaN(row[f])) sum2 += (row[f] - mean[f]) * (row[f] - mean[f]);
            std[f] = count > 1 ? Math.Sqrt(sum2 / (count - 1)) : 1.0;
            if (std[f] < 1e-9) std[f] = 1.0;
        }
        return (mean, std);
    }

    // ── Gain ranking ──────────────────────────────────────────────────────────

    private static (int Idx, float Score)[] GetGainRanking(ITransformer model, string[] featureCols)
    {
        var predictor = (RegressionPredictionTransformer<FastTreeRegressionModelParameters>)model;
        VBuffer<float> buf = default;
        predictor.Model.GetFeatureWeights(ref buf);

        return buf.DenseValues()
            .Select((score, i) => (Idx: i, Score: score))
            .OrderByDescending(x => x.Score)
            .ToArray();
    }

    // ── Walk-forward split with purge + embargo ───────────────────────────────
    //
    //  purge   : remove the last `purge` rows from training.
    //            These rows have labels (future returns) that overlap the val window.
    //            For barsAhead=1 target, purge=1.
    //
    //  embargo : skip `embargo` rows after the purge zone before val starts.
    //            These rows have features (e.g. RealizedVol_50) computed using
    //            data from the val window. Set to max feature lookback (50 bars).
    //
    //  Timeline:
    //    [─────────── train ──────────][purge][embargo][─── val ───]
    //
    //  If purge+embargo would leave val empty the fold is silently skipped.

    private static IEnumerable<(float[][] TrainF, float[] TrainL, float[][] ValF, float[] ValL)>
        WalkForwardSplit(float[][] features, float[] labels, int numFolds,
                         int purge = 1, int embargo = 50)
    {
        int n         = labels.Length;
        int chunkSize = n / (numFolds + 1);
        if (chunkSize == 0) yield break;

        for (int fold = 0; fold < numFolds; fold++)
        {
            int rawTrainEnd = (fold + 1) * chunkSize;
            int rawValEnd   = fold == numFolds - 1 ? n : (fold + 2) * chunkSize;

            // Purge: shorten training set so labels don't overlap val
            int trainEnd = Math.Max(0, rawTrainEnd - purge);

            // Embargo: skip rows whose features were computed with val data
            int valStart = rawTrainEnd + embargo;
            int valEnd   = rawValEnd;

            if (trainEnd == 0 || valStart >= valEnd) continue;

            yield return (
                features[..trainEnd],
                labels[..trainEnd],
                features[valStart..valEnd],
                labels[valStart..valEnd]
            );
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static (float[][] Train, float[][] Other) Impute(
        string imputer, float[][] trainRaw, float[][] otherRaw)
    {
        if (imputer == "knn")
        {
            var imp = new KnnImputer(k: 5, maxDistanceCols: 100);
            imp.Fit(trainRaw);
            return (imp.Transform(trainRaw), otherRaw.Length > 0 ? imp.Transform(otherRaw) : []);
        }
        else
        {
            var imp = new MeanImputer();
            imp.Fit(trainRaw);
            return (imp.Transform(trainRaw), otherRaw.Length > 0 ? imp.Transform(otherRaw) : []);
        }
    }

    internal static (float[][] Features, float[] Labels, DateTime[] Timestamps) LoadCsv(
        string path,
        int[]  featureIndices,
        int    labelIndex)
    {
        var featuresList   = new List<float[]>();
        var labelsList     = new List<float>();
        var timestampsList = new List<DateTime>();

        foreach (var line in File.ReadLines(path).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var parts = line.Split(',');

            if (labelIndex >= parts.Length || string.IsNullOrEmpty(parts[labelIndex])) continue;
            if (!float.TryParse(parts[labelIndex], NumberStyles.Float,
                    CultureInfo.InvariantCulture, out var label)) continue;
            if (float.IsNaN(label)) continue;

            // Column 0 is always Timestamp; parse it for black swan filtering.
            DateTime ts = default;
            if (parts.Length > 0)
                DateTime.TryParseExact(parts[0], "yyyy-MM-dd HH:mm:ss",
                    CultureInfo.InvariantCulture, DateTimeStyles.None, out ts);

            var features = new float[featureIndices.Length];
            for (int i = 0; i < featureIndices.Length; i++)
            {
                int fi = featureIndices[i];
                features[i] = fi < parts.Length
                    && !string.IsNullOrEmpty(parts[fi])
                    && float.TryParse(parts[fi], NumberStyles.Float,
                           CultureInfo.InvariantCulture, out var fv)
                    ? fv
                    : float.NaN;
            }

            featuresList.Add(features);
            labelsList.Add(label);
            timestampsList.Add(ts);
        }

        return (featuresList.ToArray(), labelsList.ToArray(), timestampsList.ToArray());
    }

    private static IDataView ToDataView(
        MLContext mlContext,
        float[][] features,
        float[]   labels,
        int       numFeatures)
    {
        var rows = features.Zip(labels, (f, l) => new ModelInput { Features = f, Label = l });

        var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
        schemaDef["Features"].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, numFeatures);

        return mlContext.Data.LoadFromEnumerable(rows, schemaDef);
    }

    private static void PrintTopFeatures(
        ITransformer model,
        string[]     featureCols,
        float[][]    features,
        float[]      labels,
        int          topN = 20)
    {
        var ranked = GetGainRanking(model, featureCols).Take(topN).ToArray();

        Console.WriteLine($"\n  Top {topN} features (gain):");
        Console.WriteLine($"  {"#",-4} {"Feature",-55} {"Score",10}  {"Corr",6}");
        Console.WriteLine($"  {new string('-', 82)}");
        for (int i = 0; i < ranked.Length; i++)
        {
            string name    = ranked[i].Idx < featureCols.Length ? featureCols[ranked[i].Idx] : $"f{ranked[i].Idx}";
            double corr    = PearsonCorr(features, labels, ranked[i].Idx);
            string corrStr = double.IsNaN(corr) ? "   n/a" : $"{corr:+0.000;-0.000}";
            Console.WriteLine($"  {i + 1,-4} {name,-55} {ranked[i].Score,10:F4}  {corrStr}");
        }
    }

    private static void WriteFeatureRanking(
        ITransformer model,
        string[]     featureCols,
        float[][]    features,
        float[]      labels,
        string       path)
    {
        var ranked = GetGainRanking(model, featureCols);

        using var sw = new System.IO.StreamWriter(path, append: false, System.Text.Encoding.UTF8);
        sw.WriteLine($"{"#",-5} {"Gain",10}  {"Corr",7}  Feature");
        sw.WriteLine(new string('-', 90));
        for (int i = 0; i < ranked.Length; i++)
        {
            string name    = ranked[i].Idx < featureCols.Length ? featureCols[ranked[i].Idx] : $"f{ranked[i].Idx}";
            double corr    = PearsonCorr(features, labels, ranked[i].Idx);
            string corrStr = double.IsNaN(corr) ? "    n/a" : $"{corr:+0.000;-0.000}";
            sw.WriteLine($"{i + 1,-5} {ranked[i].Score,10:F4}  {corrStr}  {name}");
        }
    }

    private static double PearsonCorr(float[][] features, float[] labels, int featureIdx)
    {
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        int n = 0;
        for (int i = 0; i < features.Length; i++)
        {
            float x = features[i][featureIdx];
            if (float.IsNaN(x)) continue;
            float y = labels[i];
            sumX += x; sumY += y; sumXY += (double)x * y;
            sumX2 += (double)x * x; sumY2 += (double)y * y;
            n++;
        }
        if (n < 2) return double.NaN;
        double meanX = sumX / n, meanY = sumY / n;
        double cov   = sumXY / n - meanX * meanY;
        double stdX  = Math.Sqrt(sumX2 / n - meanX * meanX);
        double stdY  = Math.Sqrt(sumY2 / n - meanY * meanY);
        return stdX > 1e-12 && stdY > 1e-12 ? cov / (stdX * stdY) : double.NaN;
    }

    private static IEstimator<ITransformer> BuildPipeline(MLContext mlContext) =>
        mlContext.Regression.Trainers.FastTree(
            new FastTreeRegressionTrainer.Options
            {
                LabelColumnName            = "Label",
                FeatureColumnName          = "Features",
                NumberOfLeaves             = 63,
                MinimumExampleCountPerLeaf = 20,
                LearningRate               = 0.05,
                NumberOfTrees              = 500,
            });

}
