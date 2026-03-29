using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Imputers;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Trainer;

// Schema used for ML.NET DataView — Features vector + Label scalar
public sealed class ModelInput
{
    public float[] Features { get; set; } = [];
    public float   Label    { get; set; }
}

public static class ModelTrainer
{
    // All four NaN-rate buckets produced by the Transformer.
    // Imputer: KNN for ≤50% NaN (meaningful neighbours); column-mean for >50% NaN
    // (KNN degenerates to mean when shared column overlap is near zero — use mean
    // directly for speed and honesty).
    private static readonly (string Suffix, string Label, string Imputer)[] TrainBuckets =
    [
        ("nan_0_66",   "≤66% NaN",  "knn"),
        ("nan_66_100", ">66% NaN",  "mean"),
    ];

    public static void Run(string datasetDir, string modelDir, int numFolds = 10)
    {
        var mlContext = new MLContext(seed: 42);

        // Discover bucket CSVs — pattern: xauusd_4h_dataset_nan_*.csv
        var bucketCsvs = TrainBuckets
            .Select(b => (
                b.Label,
                b.Suffix,
                b.Imputer,
                Path   : Path.Combine(datasetDir, $"xauusd_4h_dataset_{b.Suffix}.csv")))
            .ToArray();

        // Validate at least one bucket exists
        var missing = bucketCsvs.Where(b => !File.Exists(b.Path)).ToArray();
        if (missing.Length == bucketCsvs.Length)
        {
            Console.WriteLine("ERROR: No NaN-bucket CSVs found. Run the Importer first.");
            Console.WriteLine($"  Expected files like: {bucketCsvs[0].Path}");
            return;
        }
        foreach (var m in missing)
            Console.WriteLine($"[WARN] Bucket CSV not found, skipping: {m.Path}");

        // Discover target columns from the first available bucket CSV
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

        // ── Train one model per (target, bucket) ──────────────────────────────

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

                // Read schema from this bucket's CSV (feature set may differ after pruning)
                var cols       = File.ReadLines(csvPath).First().Split(',');
                var featureCols = cols
                    .Skip(1)
                    .Where(c => !c.Contains("_Target_"))
                    .ToArray();
                var colIndex   = cols.Select((n, i) => (n, i)).ToDictionary(x => x.n, x => x.i);

                if (!colIndex.TryGetValue(targetCol, out var targetIdx))
                {
                    Console.WriteLine($"  [SKIP] Target column not found in {suffix} CSV.");
                    continue;
                }

                var featureIdx = featureCols.Select(c => colIndex[c]).ToArray();

                Console.WriteLine($"  Features : {featureCols.Length}");

                var sw = Stopwatch.StartNew();

                var (rawF, labels) = LoadCsv(csvPath, featureIdx, targetIdx);
                if (labels.Length == 0)
                {
                    Console.WriteLine($"  [SKIP] No labelled rows in {suffix} CSV.");
                    continue;
                }

                // ── Hold-out split (last chunk = most recent data) ────────────
                double holdoutPct = labels.Length < 10_000 ? 0.30 : 0.20;
                int chunkSize  = (int)(labels.Length * holdoutPct);
                int cvSize     = labels.Length - chunkSize;
                var cvRawF     = chunkSize > 0 ? rawF[..cvSize]     : rawF;
                var cvLabels   = chunkSize > 0 ? labels[..cvSize]   : labels;
                var hoRawF     = chunkSize > 0 ? rawF[cvSize..]     : [];
                var hoLabels   = chunkSize > 0 ? labels[cvSize..]   : [];

                Console.WriteLine($"  Rows     : {labels.Length}  (CV={cvLabels.Length}  hold-out={hoLabels.Length})");

                // ── Walk-forward cross-validation ─────────────────────────────
                Console.WriteLine($"\n  Walk-forward CV ({numFolds} folds):");

                var cvRmse   = new List<double>();
                var cvMae    = new List<double>();
                var cvR2     = new List<double>();
                var foldRows = new List<(int FoldNum, int TrainLen, int ValLen)>();

                int foldNum = 0;
                foreach (var (rawTrainF, trainL, rawValF, valL) in WalkForwardSplit(cvRawF, cvLabels, numFolds))
                {
                    foldNum++;
                    Console.WriteLine($"\n  ── Fold {foldNum}/{numFolds}  train={trainL.Length}  val={valL.Length} ──");

                    float[][] fTrainF, fValF;
                    if (imputer == "knn")
                    {
                        var imp = new KnnImputer(k: 5, maxDistanceCols: 100);
                        imp.Fit(rawTrainF);
                        fTrainF = imp.Transform(rawTrainF);
                        fValF   = imp.Transform(rawValF);
                    }
                    else
                    {
                        var imp = new MeanImputer();
                        imp.Fit(rawTrainF);
                        fTrainF = imp.Transform(rawTrainF);
                        fValF   = imp.Transform(rawValF);
                    }

                    var trainDv   = ToDataView(mlContext, fTrainF, trainL, featureCols.Length);
                    var foldModel = BuildPipeline(mlContext).Fit(trainDv);

                    PrintTopFeatures(foldModel, featureCols, fTrainF, trainL, topN: 30);

                    var valDv    = ToDataView(mlContext, fValF, valL, featureCols.Length);
                    var foldPred = foldModel.Transform(valDv);
                    var fm       = mlContext.Regression.Evaluate(foldPred, labelColumnName: "Label");

                    Console.WriteLine($"\n  RMSE={fm.RootMeanSquaredError:F6}  MAE={fm.MeanAbsoluteError:F6}  R²={fm.RSquared:F4}");

                    cvRmse.Add(fm.RootMeanSquaredError);
                    cvMae.Add(fm.MeanAbsoluteError);
                    cvR2.Add(fm.RSquared);
                    foldRows.Add((foldNum, trainL.Length, valL.Length));
                }

                // ── Fold stats table ──────────────────────────────────────────
                Console.WriteLine($"\n  {"Fold",-6} {"Train",6} {"Val",5}  {"RMSE",10}  {"MAE",10}  {"R²",7}");
                Console.WriteLine($"  {new string('-', 55)}");
                for (int fi = 0; fi < foldRows.Count; fi++)
                {
                    var (fn, tLen, vLen) = foldRows[fi];
                    Console.WriteLine($"  {fn,-6} {tLen,6} {vLen,5}  " +
                                      $"{cvRmse[fi],10:F6}  {cvMae[fi],10:F6}  {cvR2[fi],7:F4}");
                }

                if (cvRmse.Count > 0)
                {
                    double mRmse  = cvRmse.Average();
                    double sdRmse = Math.Sqrt(cvRmse.Average(v => (v - mRmse) * (v - mRmse)));
                    Console.WriteLine($"  {"mean",-6} {"",6} {"",5}  {mRmse,10:F6}  {cvMae.Average(),10:F6}  {cvR2.Average(),7:F4}  ±{sdRmse:F6}");
                }

                // ── Train final model on CV portion ───────────────────────────
                Console.WriteLine($"\n  Final model:");
                float[][] features;
                float[][] hoFeatures = [];
                if (imputer == "knn")
                {
                    var imp = new KnnImputer(k: 5, maxDistanceCols: 100);
                    imp.Fit(cvRawF);
                    features   = imp.Transform(cvRawF);
                    if (hoRawF.Length > 0) hoFeatures = imp.Transform(hoRawF);
                }
                else
                {
                    var imp = new MeanImputer();
                    imp.Fit(cvRawF);
                    features   = imp.Transform(cvRawF);
                    if (hoRawF.Length > 0) hoFeatures = imp.Transform(hoRawF);
                }

                var dataView     = ToDataView(mlContext, features, cvLabels, featureCols.Length);
                var trainedModel = BuildPipeline(mlContext).Fit(dataView);
                sw.Stop();

                // Hold-out validation
                if (hoFeatures.Length > 0)
                {
                    var hoDv      = ToDataView(mlContext, hoFeatures, hoLabels, featureCols.Length);
                    var hoPred    = trainedModel.Transform(hoDv);
                    var hoMetrics = mlContext.Regression.Evaluate(hoPred, labelColumnName: "Label");
                    Console.WriteLine($"  Hold-out  RMSE={hoMetrics.RootMeanSquaredError:F6}  " +
                                      $"MAE={hoMetrics.MeanAbsoluteError:F6}  " +
                                      $"R²={hoMetrics.RSquared:F4}  ({sw.Elapsed.TotalSeconds:F1}s)");
                }
                else
                {
                    Console.WriteLine($"  (no hold-out rows — dataset too small for {numFolds} folds)  ({sw.Elapsed.TotalSeconds:F1}s)");
                }

                PrintTopFeatures(trainedModel, featureCols, features, cvLabels);

                // Save model + sidecar
                var modelName = $"{targetCol}_{suffix}";
                var modelPath = Path.Combine(modelDir, $"{modelName}.zip");
                mlContext.Model.Save(trainedModel, dataView.Schema, modelPath);
                Console.WriteLine($"  Saved → {modelPath}");

                var metaPath = Path.ChangeExtension(modelPath, ".features.json");
                File.WriteAllText(metaPath, JsonSerializer.Serialize(
                    new ModelMeta
                    {
                        TargetColumn   = targetCol,
                        NanBucket      = suffix,
                        Imputer        = imputer,
                        FeatureColumns = featureCols
                    },
                    new JsonSerializerOptions { WriteIndented = true }));
                Console.WriteLine($"  Saved → {metaPath}");
            }
        }

        Console.WriteLine($"\n{new string('=', 70)}");
        Console.WriteLine("Training complete.");
    }

    // ── Walk-forward split ────────────────────────────────────────────────────

    /// <summary>
    /// Expanding-window walk-forward split. Divides data into (numFolds+1) equal chunks.
    /// Fold i trains on chunks 0..i and validates on chunk i+1.
    /// Rows are assumed to be in chronological order.
    /// </summary>
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

    /// <summary>
    /// Reads a CSV, returning a flat float[][] (features) and float[] (labels).
    /// Rows where the label is missing or NaN are skipped.
    /// Feature NaNs are preserved for the imputer to fill.
    /// </summary>
    internal static (float[][] Features, float[] Labels) LoadCsv(
        string path,
        int[]  featureIndices,
        int    labelIndex)
    {
        var featuresList = new List<float[]>();
        var labelsList   = new List<float>();

        foreach (var line in File.ReadLines(path).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var parts = line.Split(',');

            if (labelIndex >= parts.Length || string.IsNullOrEmpty(parts[labelIndex])) continue;
            if (!float.TryParse(parts[labelIndex], NumberStyles.Float,
                    CultureInfo.InvariantCulture, out var label)) continue;
            if (float.IsNaN(label)) continue;

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
        }

        return (featuresList.ToArray(), labelsList.ToArray());
    }

    /// <summary>
    /// Reads only the date column (col 0) from a CSV, skipping the header.
    /// </summary>
    private static string[] LoadDates(string path)
    {
        var dates = new List<string>();
        foreach (var line in File.ReadLines(path).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var comma = line.IndexOf(',');
            dates.Add(comma > 0 ? line[..comma] : line);
        }
        return dates.ToArray();
    }

    /// <summary>
    /// Prints date range, target distribution, and EMA/ATR feature summary for a fold.
    /// </summary>
    private static void PrintFoldDiagnostics(
        string    foldName,
        string    valFile,
        float[][] rawTrainF,
        float[][] rawValF,
        float[]   valL,
        string[]  featureCols)
    {
        // ── Date range ────────────────────────────────────────────────────────
        var dates = LoadDates(valFile);
        var dateRange = dates.Length > 0
            ? $"{dates[0]} → {dates[^1]}"
            : "n/a";

        // ── Target distribution ────────────────────────────────────────────────
        double tMean = valL.Average();
        double tVar  = valL.Average(v => (v - tMean) * (v - tMean));
        double tStd  = Math.Sqrt(tVar);

        Console.WriteLine($"\n  [{foldName}]  {dateRange}  ({valL.Length} rows)");
        Console.WriteLine($"    Target : mean={tMean,+10:F6}  std={tStd:F6}  " +
                          $"min={valL.Min(),+10:F6}  max={valL.Max(),+10:F6}");

        // ── NaN stats ─────────────────────────────────────────────────────────
        static (double rate, int rowsAffected, int fullNanCols, (string col, double pct)[] top) NanStats(
            float[][] data, string[] cols, int topN = 5)
        {
            if (data.Length == 0) return (0, 0, 0, []);
            int nRows = data.Length, nCols = data[0].Length;
            long totalCells = (long)nRows * nCols;
            long nanCells   = 0;
            int  rowsAff    = 0;
            var  colNans    = new int[nCols];

            foreach (var row in data)
            {
                bool rowHasNan = false;
                for (int c = 0; c < nCols; c++)
                {
                    if (float.IsNaN(row[c])) { nanCells++; colNans[c]++; rowHasNan = true; }
                }
                if (rowHasNan) rowsAff++;
            }

            int fullNan = colNans.Count(n => n == nRows);
            var top = colNans
                .Select((n, i) => (col: i < cols.Length ? cols[i] : $"col{i}", pct: (double)n / nRows * 100))
                .Where(x => x.pct > 0)
                .OrderByDescending(x => x.pct)
                .Take(topN)
                .ToArray();

            return ((double)nanCells / totalCells * 100, rowsAff, fullNan, top);
        }

        var (vRate, vRows, vFull, vTop) = NanStats(rawValF,   featureCols);
        var (tRate, tRows, tFull, _)    = NanStats(rawTrainF, featureCols);

        Console.WriteLine($"    NaN val  : {vRate,5:F1}% of cells  " +
                          $"{vRows}/{rawValF.Length} rows affected  {vFull} fully-NaN cols");
        Console.WriteLine($"    NaN train: {tRate,5:F1}% of cells  " +
                          $"{tRows}/{rawTrainF.Length} rows affected  {tFull} fully-NaN cols");

        if (vTop.Length > 0)
        {
            var topStr = string.Join("  ", vTop.Select(x => $"{x.col}({x.pct:F0}%)"));
            Console.WriteLine($"    Top NaN  : {topStr}");
        }

        // ── EMA / ATR feature summary (XAU/USD 4h) ────────────────────────────
        static void PrintFeatureStat(string label, int localIdx, float[][] data)
        {
            if (localIdx < 0) { Console.WriteLine($"    {label,-30} not found"); return; }
            var vals = data.Select(r => r[localIdx]).Where(v => !float.IsNaN(v)).ToArray();
            if (vals.Length == 0) { Console.WriteLine($"    {label,-30} all NaN"); return; }
            double mean = vals.Average();
            double std  = Math.Sqrt(vals.Average(v => (v - mean) * (v - mean)));
            Console.WriteLine($"    {label,-30} mean={mean,+12:F4}  std={std:F4}  ({vals.Length} non-NaN)");
        }

        int emaIdx = Array.FindIndex(featureCols, c =>
            c.StartsWith("XAUUSD_4h_EMA_", StringComparison.OrdinalIgnoreCase));
        int atrIdx = Array.FindIndex(featureCols, c =>
            c.StartsWith("XAUUSD_4h_ATR_", StringComparison.OrdinalIgnoreCase));

        PrintFeatureStat(emaIdx >= 0 ? featureCols[emaIdx] : "XAUUSD_4h_EMA_*", emaIdx, rawValF);
        PrintFeatureStat(atrIdx >= 0 ? featureCols[atrIdx] : "XAUUSD_4h_ATR_*", atrIdx, rawValF);
    }

    /// <summary>
    /// Creates an ML.NET IDataView from imputed feature arrays and labels.
    /// The Features column is declared as a fixed-size float vector so
    /// FastTree can determine the feature count from the schema.
    /// </summary>
    private static IDataView ToDataView(
        MLContext mlContext,
        float[][] features,
        float[]   labels,
        int       numFeatures)
    {
        var rows = features
            .Zip(labels, (f, l) => new ModelInput { Features = f, Label = l });

        // Declare Features as a fixed-size vector in the schema
        var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
        schemaDef["Features"].ColumnType =
            new VectorDataViewType(NumberDataViewType.Single, numFeatures);

        return mlContext.Data.LoadFromEnumerable(rows, schemaDef);
    }

    private static void PrintTopFeatures(
        ITransformer model,
        string[]    featureCols,
        float[][]   features,
        float[]     labels,
        int         topN = 20)
    {
        var predictor = (RegressionPredictionTransformer<FastTreeRegressionModelParameters>)model;
        VBuffer<float> importanceBuffer = default;
        predictor.Model.GetFeatureWeights(ref importanceBuffer);

        var top = importanceBuffer.DenseValues()
            .Select((score, i) => (Name: i < featureCols.Length ? featureCols[i] : $"f{i}", Score: score, Idx: i))
            .OrderByDescending(x => x.Score)
            .Take(topN)
            .ToArray();

        Console.WriteLine($"\n  Top {topN} features (gain):");
        Console.WriteLine($"  {"#",-4} {"Feature",-55} {"Score",10}  {"Corr",6}");
        Console.WriteLine($"  {new string('-', 82)}");
        for (int i = 0; i < top.Length; i++)
        {
            double corr = PearsonCorr(features, labels, top[i].Idx);
            string corrStr = double.IsNaN(corr) ? "   n/a" : $"{corr:+0.000;-0.000}";
            Console.WriteLine($"  {i + 1,-4} {top[i].Name,-55} {top[i].Score,10:F4}  {corrStr}");
        }
    }

    /// <summary>
    /// Pearson correlation between a single feature column and the label vector.
    /// Rows where the feature is NaN are excluded.
    /// </summary>
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
        double cov  = sumXY / n - meanX * meanY;
        double stdX = Math.Sqrt(sumX2 / n - meanX * meanX);
        double stdY = Math.Sqrt(sumY2 / n - meanY * meanY);
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
