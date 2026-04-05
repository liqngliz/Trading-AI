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
    private const double Pass3FeatureFraction = 0.20;
    private const int    Pass4FeatureCount    = 20;
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

        // ── Recent macro shocks ────────────────────────────────────────────────
        // Liberation Day: US tariff shock Apr 2 2026; global equity/gold vol spike
        (new(2026,  3, 19), new(2026,  4,  9), "Liberation Day Tariff Shock",       null),
    ];

    // targetPrefix: the safe-symbol prefix of the target column, e.g. "XAUUSD"
    private static bool IsNearBlackSwan(DateTime ts, string targetPrefix) =>
        BlackSwanPeriods.Any(p =>
            ts >= p.From && ts <= p.To &&
            (p.Assets is null || p.Assets.Any(a => targetPrefix.StartsWith(a, StringComparison.Ordinal))));

    public static void Run(string datasetDir, string modelDir, int numFolds = 10, bool retrain = false)
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
                var featureCols = cols.Skip(1).Where(c => !c.Contains("_Target_")).ToArray();
                var colIndex    = cols.Select((n, i) => (n, i)).ToDictionary(x => x.n, x => x.i);

                if (!colIndex.TryGetValue(targetCol, out var targetIdx))
                {
                    Console.WriteLine($"  [SKIP] Target column not found in {suffix} CSV.");
                    continue;
                }

                var featureIdx = featureCols.Select(c => colIndex[c]).ToArray();
                Console.WriteLine($"  Features : {featureCols.Length}");

                var sw = Stopwatch.StartNew();

                // Shuffle all rows (Fisher-Yates, seed 42) then load from shuffled CSV.
                // FastTree is order-agnostic; shuffling ensures the hold-out is a random
                // mix of the dataset rather than the temporal tail.
                var shuffledCsvPath = ShuffleCsvAndSave(csvPath, seed: 42, reuseExisting: retrain);
                var (rawF, labels, timestamps) = LoadCsv(shuffledCsvPath, featureIdx, targetIdx);
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

                // ── Hold-out split (random, not temporal) ────────────────────
                // Rows are already shuffled, so slicing [cvSize..] gives a
                // random 30 % of the full dataset — not the temporal last 30 %.
                double holdoutPct = labels.Length < 10_000 ? 0.30 : 0.20;
                int    hoSize     = (int)(labels.Length * holdoutPct);
                int    cvSize     = labels.Length - hoSize;
                var    cvRawF     = hoSize > 0 ? rawF[..cvSize]   : rawF;
                var    cvLabels   = hoSize > 0 ? labels[..cvSize] : labels;
                var    hoRawF     = hoSize > 0 ? rawF[cvSize..]   : [];
                var    hoLabels   = hoSize > 0 ? labels[cvSize..] : [];
                var    hoTs       = hoSize > 0 ? timestamps[cvSize..] : [];

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

                // ── Helper: slice raw feature arrays to a column subset ───────
                float[][] SliceCols(float[][] data, int[] idx) =>
                    data.Select(row => idx.Select(i => row[i]).ToArray()).ToArray();

                int[] ColIndices(HashSet<int> keep) =>
                    featureCols.Select((_, i) => i).Where(keep.Contains).ToArray();

                // ── Pass 1: all features ──────────────────────────────────────
                S($"\npass 1 ({featureCols.Length} features):");
                var cvLines1 = RunCV(mlContext, imputer, cvRawF, cvLabels, featureCols, numFolds);
                foreach (var l in cvLines1) S(l);

                S($"\n  ── Pass 1 final model ──");
                var (model1, hoRmse1, cvF1, hoF1) = TrainFinalModel(
                    mlContext, imputer, cvRawF, cvLabels, hoRawF, hoLabels, featureCols);
                var (p1Ho, _) = FormatHoldOut(hoLabels, mlContext, model1, hoF1, featureCols.Length, sw);
                Console.WriteLine(p1Ho); summary.Add(p1Ho);

                // ── Derive feature subsets from pass 1 gain ranking ───────────
                var gainRanking = GetGainRanking(model1, featureCols);

                int keepN2       = Math.Max(1, (int)Math.Ceiling(featureCols.Length * Pass2FeatureFraction));
                var topIdx2      = gainRanking.Take(keepN2).Select(x => x.Idx).ToHashSet();
                var featureCols2 = featureCols.Where((_, i) => topIdx2.Contains(i)).ToArray();
                var localIdx2    = ColIndices(topIdx2);
                var cvRawF2      = SliceCols(cvRawF, localIdx2);
                var hoRawF2      = SliceCols(hoRawF, localIdx2);

                int keepN3       = Math.Max(1, (int)Math.Ceiling(featureCols.Length * Pass3FeatureFraction));
                var topIdx3      = gainRanking.Take(keepN3).Select(x => x.Idx).ToHashSet();
                var featureCols3 = featureCols.Where((_, i) => topIdx3.Contains(i)).ToArray();
                var localIdx3    = ColIndices(topIdx3);
                var cvRawF3      = SliceCols(cvRawF, localIdx3);
                var hoRawF3      = SliceCols(hoRawF, localIdx3);

                int keepN4       = Math.Min(Pass4FeatureCount, featureCols.Length);
                var topIdx4      = gainRanking.Take(keepN4).Select(x => x.Idx).ToHashSet();
                var featureCols4 = featureCols.Where((_, i) => topIdx4.Contains(i)).ToArray();
                var localIdx4    = ColIndices(topIdx4);
                var cvRawF4      = SliceCols(cvRawF, localIdx4);
                var hoRawF4      = SliceCols(hoRawF, localIdx4);

                // ── Pass 2: top 40% ───────────────────────────────────────────
                S($"\npass 2 (top {keepN2}/{featureCols.Length} = 40% features):");
                var cvLines2 = RunCV(mlContext, imputer, cvRawF2, cvLabels, featureCols2, numFolds);
                foreach (var l in cvLines2) S(l);

                S($"\n  ── Pass 2 final model ──");
                var (model2, hoRmse2, cvF2, hoF2) = TrainFinalModel(
                    mlContext, imputer, cvRawF2, cvLabels, hoRawF2, hoLabels, featureCols2);
                var (p2Ho, _) = FormatHoldOut(hoLabels, mlContext, model2, hoF2, featureCols2.Length, sw);
                Console.WriteLine(p2Ho); summary.Add(p2Ho);

                // ── Pass 3: top 20% ───────────────────────────────────────────
                S($"\npass 3 (top {keepN3}/{featureCols.Length} = 20% features):");
                var cvLines3 = RunCV(mlContext, imputer, cvRawF3, cvLabels, featureCols3, numFolds);
                foreach (var l in cvLines3) S(l);

                S($"\n  ── Pass 3 final model ──");
                var (model3, hoRmse3, cvF3, hoF3) = TrainFinalModel(
                    mlContext, imputer, cvRawF3, cvLabels, hoRawF3, hoLabels, featureCols3);
                var (p3Ho, _) = FormatHoldOut(hoLabels, mlContext, model3, hoF3, featureCols3.Length, sw);
                Console.WriteLine(p3Ho); summary.Add(p3Ho);

                // ── Pass 4: top 20 features ───────────────────────────────────
                S($"\npass 4 (top {keepN4} features):");
                var cvLines4 = RunCV(mlContext, imputer, cvRawF4, cvLabels, featureCols4, numFolds);
                foreach (var l in cvLines4) S(l);

                S($"\n  ── Pass 4 final model ──");
                var (model4, hoRmse4, cvF4, hoF4) = TrainFinalModel(
                    mlContext, imputer, cvRawF4, cvLabels, hoRawF4, hoLabels, featureCols4);
                var (p4Ho, _) = FormatHoldOut(hoLabels, mlContext, model4, hoF4, featureCols4.Length, sw);
                Console.WriteLine(p4Ho); summary.Add(p4Ho);

                // ── Pick winner ───────────────────────────────────────────────
                double bestRmse = Math.Min(hoRmse1, Math.Min(hoRmse2, Math.Min(hoRmse3, hoRmse4)));
                var winModel = bestRmse == hoRmse4 ? model4       : bestRmse == hoRmse3 ? model3       : bestRmse == hoRmse2 ? model2       : model1;
                var winCols  = bestRmse == hoRmse4 ? featureCols4 : bestRmse == hoRmse3 ? featureCols3 : bestRmse == hoRmse2 ? featureCols2 : featureCols;
                var winCvF   = bestRmse == hoRmse4 ? cvF4         : bestRmse == hoRmse3 ? cvF3         : bestRmse == hoRmse2 ? cvF2         : cvF1;
                var winPass  = bestRmse == hoRmse4 ? "pass 4"     : bestRmse == hoRmse3 ? "pass 3"     : bestRmse == hoRmse2 ? "pass 2"     : "pass 1";

                S($"\n  Winner: {winPass}  (hold-out RMSE pass1={hoRmse1:F6}  pass2={hoRmse2:F6}  pass3={hoRmse3:F6}  pass4={hoRmse4:F6})");

                // ── Clean hold-out evaluation (winner model) ──────────────────
                var winHoLocalIdx = bestRmse == hoRmse4 ? localIdx4
                                  : bestRmse == hoRmse3 ? localIdx3
                                  : bestRmse == hoRmse2 ? localIdx2
                                  : Enumerable.Range(0, featureCols.Length).ToArray();
                var cleanHoSliced = cleanHoRawF.Length > 0
                    ? cleanHoRawF.Select(row => winHoLocalIdx.Select(i => row[i]).ToArray()).ToArray()
                    : [];

                S($"\n── Clean hold-out (black-swan rows excluded: {blackSwanCount}) ──");
                if (cleanHoLabels.Length >= 10)
                {
                    var (cleanHoLine, cleanRmse) = FormatHoldOut(cleanHoLabels, mlContext, winModel,
                        Impute(imputer, hoRawF.Select(row => winHoLocalIdx.Select(i => row[i]).ToArray()).ToArray(),
                               cleanHoSliced).Other,
                        winCols.Length, sw);
                    S(cleanHoLine);

                    // ── Black swan sensitivity warning ────────────────────────
                    // fullRmse / cleanRmse: measures how much tail events inflate the hold-out error.
                    //   < 1.2  → normal
                    //   1.2–1.5 → mild (model struggles on extremes)
                    //   > 1.5  → WARNING (strong tail sensitivity)
                    //   > 2.0  → STRONG WARNING (model score dominated by tail events)
                    if (!double.IsNaN(cleanRmse) && cleanRmse > 0 && !double.IsNaN(bestRmse))
                    {
                        double ratio = bestRmse / cleanRmse;
                        string ratioStr = $"full/clean RMSE ratio = {ratio:F3}  ({bestRmse:F6} / {cleanRmse:F6})";
                        if (ratio > 2.0)
                            S($"\n  *** STRONG WARNING: {ratioStr}");
                        else if (ratio > 1.5)
                            S($"\n  ** WARNING: {ratioStr}");
                        else if (ratio > 1.2)
                            S($"\n  * MILD WARNING: {ratioStr}");
                        else
                            S($"\n  OK: {ratioStr}");
                    }
                }
                else
                {
                    S($"  (too few clean rows: {cleanHoLabels.Length} — need ≥ 10)");
                }

                // ── Temporal hold-out (last 30% of original time-ordered data) ─
                // Load the original (unshuffled) bucket CSV and take the last
                // hoSize rows by time — gives a pure out-of-sample future window.
                var (tempAllF, tempAllL, tempAllTs) = LoadCsv(csvPath, featureIdx, targetIdx);
                int tempHoStart   = Math.Max(0, tempAllL.Length - hoSize);
                var tempHoRawSliced = SliceCols(tempAllF[tempHoStart..], winHoLocalIdx);
                var tempHoLabels    = tempAllL[tempHoStart..];
                var winCvRawSliced  = SliceCols(cvRawF, winHoLocalIdx);

                S($"\n── Temporal hold-out (last {holdoutPct:P0} by time, {tempHoLabels.Length} rows) ──");
                if (tempHoLabels.Length >= 10)
                {
                    var (tempHoLine, _) = FormatHoldOut(tempHoLabels, mlContext, winModel,
                        Impute(imputer, winCvRawSliced, tempHoRawSliced).Other,
                        winCols.Length, sw);
                    S(tempHoLine);
                }
                else
                {
                    S($"  (too few rows: {tempHoLabels.Length} — need ≥ 10)");
                }

                // ── Black-swan hold-out (rows near BS events incl. Liberation Day)
                // Evaluates model performance specifically during tail/crisis periods.
                var bsIdx      = Enumerable.Range(0, tempAllL.Length)
                    .Where(i => tempAllTs.Length > i && IsNearBlackSwan(tempAllTs[i], targetPrefix))
                    .ToArray();
                var bsHoRawSliced = SliceCols([.. bsIdx.Select(i => tempAllF[i])], winHoLocalIdx);
                var bsHoLabels    = bsIdx.Select(i => tempAllL[i]).ToArray();

                S($"\n── Black-swan hold-out ({bsHoLabels.Length} rows near BS events) ──");
                if (bsHoLabels.Length >= 10)
                {
                    var (bsHoLine, _) = FormatHoldOut(bsHoLabels, mlContext, winModel,
                        Impute(imputer, winCvRawSliced, bsHoRawSliced).Other,
                        winCols.Length, sw);
                    S(bsHoLine);
                }
                else
                {
                    S($"  (too few rows: {bsHoLabels.Length} — need ≥ 10)");
                }

                sw.Stop();

                // ── Top 30 features of winner ─────────────────────────────────
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

                // ── Save winner ───────────────────────────────────────────────
                var modelName = $"{targetCol}_{suffix}";
                var modelPath = Path.Combine(modelDir, $"{modelName}.zip");
                var winDv     = ToDataView(mlContext, winCvF, cvLabels, winCols.Length);
                mlContext.Model.Save(winModel, winDv.Schema, modelPath);
                Console.WriteLine($"  Saved → {modelPath}");

                var metaPath = Path.ChangeExtension(modelPath, ".features.json");
                File.WriteAllText(metaPath, JsonSerializer.Serialize(
                    new ModelMeta
                    {
                        TargetColumn   = targetCol,
                        NanBucket      = suffix,
                        Imputer        = imputer,
                        FeatureColumns = winCols
                    },
                    new JsonSerializerOptions { WriteIndented = true }));
                Console.WriteLine($"  Saved → {metaPath}");

                var rankPath = Path.ChangeExtension(modelPath, ".feature_ranking.txt");
                WriteFeatureRanking(winModel, winCols, winCvF, cvLabels, rankPath);
                Console.WriteLine($"  Saved → {rankPath}");

                var summaryPath = Path.ChangeExtension(modelPath, ".training_summary.txt");
                File.WriteAllLines(summaryPath, summary);
                Console.WriteLine($"  Saved → {summaryPath}");
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
        int        numFolds)
    {
        var lines    = new List<string>();
        var cvRmse   = new List<double>();
        var cvMae    = new List<double>();
        var cvR2     = new List<double>();
        var foldRows = new List<(int FoldNum, int TrainLen, int ValLen)>();
        int foldNum  = 0;

        foreach (var (rawTrainF, trainL, rawValF, valL) in WalkForwardSplit(cvRawF, cvLabels, numFolds))
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
        Stopwatch    sw)
    {
        if (hoLabels.Length > 0 && hoF.Length > 0)
        {
            var hoDv    = ToDataView(mlContext, hoF, hoLabels, numFeatures);
            var metrics = mlContext.Regression.Evaluate(model.Transform(hoDv), labelColumnName: "Label");
            return (
                $"  Hold-out  RMSE={metrics.RootMeanSquaredError:F6}  " +
                $"MAE={metrics.MeanAbsoluteError:F6}  " +
                $"R²={metrics.RSquared:F4}  ({sw.Elapsed.TotalSeconds:F1}s)",
                metrics.RootMeanSquaredError);
        }
        return ($"  (no hold-out rows)  ({sw.Elapsed.TotalSeconds:F1}s)", double.NaN);
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

    // ── Walk-forward split ────────────────────────────────────────────────────

    private static IEnumerable<(float[][] TrainF, float[] TrainL, float[][] ValF, float[] ValL)>
        WalkForwardSplit(float[][] features, float[] labels, int numFolds)
    {
        int n         = labels.Length;
        int chunkSize = n / (numFolds + 1);
        if (chunkSize == 0) yield break;

        for (int fold = 0; fold < numFolds; fold++)
        {
            int trainEnd = (fold + 1) * chunkSize;
            int valStart = trainEnd;
            int valEnd   = fold == numFolds - 1 ? n : (fold + 2) * chunkSize;
            if (valStart >= valEnd) yield break;

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

    // Shuffles all labeled rows in csvPath (Fisher-Yates, fixed seed) and writes
    // the result to <csvPath without .csv>_shuffled.csv. Returns the shuffled path.
    private static string ShuffleCsvAndSave(string csvPath, int seed, bool reuseExisting = false)
    {
        var shuffledPath = Path.ChangeExtension(csvPath, null) + "_shuffled.csv";
        if (reuseExisting && File.Exists(shuffledPath))
        {
            Console.WriteLine($"  [retrain] Reusing existing shuffled CSV: {Path.GetFileName(shuffledPath)}");
            return shuffledPath;
        }

        var allLines  = File.ReadAllLines(csvPath);
        var header    = allLines[0];
        var dataLines = allLines.Skip(1).Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();

        var rng = new Random(seed);
        for (int i = dataLines.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (dataLines[i], dataLines[j]) = (dataLines[j], dataLines[i]);
        }

        using var sw = new StreamWriter(shuffledPath, append: false, System.Text.Encoding.UTF8);
        sw.WriteLine(header);
        foreach (var line in dataLines)
            sw.WriteLine(line);

        return shuffledPath;
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
