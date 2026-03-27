using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace Predictor;

// Schema used for ML.NET DataView — Features vector + Label scalar
public sealed class ModelInput
{
    public float[] Features { get; set; } = [];
    public float   Label    { get; set; }
}

public static class Trainer
{
    public static void Run(string datasetDir, string modelDir)
    {
        var fullCsvPath = Path.Combine(datasetDir, "xauusd_4h_dataset.csv");
        if (!File.Exists(fullCsvPath))
        {
            Console.WriteLine($"ERROR: Dataset file not found: {fullCsvPath}");
            return;
        }

        var mlContext = new MLContext(seed: 42);

        // ── Discover schema ────────────────────────────────────────────────────

        var allCols = File.ReadLines(fullCsvPath).First().Split(',');

        var featureCols = allCols
            .Skip(1)
            .Where(c => !c.Contains("_Target_"))
            .ToArray();

        var targetCols = allCols
            .Where(c => c.Contains("_Target_") && c.EndsWith("_Return"))
            .ToArray();

        Console.WriteLine($"Features : {featureCols.Length}");
        Console.WriteLine($"Targets  : {string.Join(", ", targetCols)}");

        var colIndex   = allCols.Select((n, i) => (n, i)).ToDictionary(x => x.n, x => x.i);
        var featureIdx = featureCols.Select(c => colIndex[c]).ToArray();

        var foldTrainFiles = Directory.GetFiles(datasetDir, "fold_*_train.csv")
            .OrderBy(f =>
            {
                var m = Path.GetFileName(f).AsSpan();
                int start = m.IndexOfAnyInRange('0', '9');
                if (start < 0) return 0;
                int end = start;
                while (end < m.Length && char.IsAsciiDigit(m[end])) end++;
                return int.TryParse(m[start..end], out var n) ? n : 0;
            })
            .ToArray();

        Console.WriteLine($"Folds    : {foldTrainFiles.Length}");

        // ── Train one model per target horizon ────────────────────────────────

        foreach (var targetCol in targetCols)
        {
            Console.WriteLine($"\n{new string('=', 60)}");
            Console.WriteLine($"Target: {targetCol}");
            Console.WriteLine(new string('=', 60));

            int targetIdx = colIndex[targetCol];

            // Walk-forward cross-validation
            if (foldTrainFiles.Length > 0)
            {
                Console.WriteLine("\nWalk-forward CV:");

                foreach (var trainFile in foldTrainFiles)
                {
                    var valFile = trainFile.Replace("_train.csv", "_val.csv");
                    if (!File.Exists(valFile)) continue;

                    var foldName = Path.GetFileNameWithoutExtension(trainFile)
                        .Replace("_train", "");

                    var (rawTrainF, trainL) = LoadCsv(trainFile, featureIdx, targetIdx);
                    var (rawValF,   valL)   = LoadCsv(valFile,   featureIdx, targetIdx);
                    if (trainL.Length == 0 || valL.Length == 0) continue;

                    PrintFoldDiagnostics(foldName, valFile, rawTrainF, rawValF, valL, featureCols);

                    Console.WriteLine($"  {"Fold",-10} {"RMSE",10} {"MAE",10} {"R²",8} {"Rows",8}");
                    Console.WriteLine($"  {new string('-', 50)}");

                    var sw = Stopwatch.StartNew();

                    // Fit imputer on training fold only — apply to both sets
                    var imputer = new KnnImputer(k: 5, maxDistanceCols: 100);
                    imputer.Fit(rawTrainF);
                    var trainF = imputer.Transform(rawTrainF);
                    var valF   = imputer.Transform(rawValF);

                    var trainView = ToDataView(mlContext, trainF, trainL, featureCols.Length);
                    var valView   = ToDataView(mlContext, valF,   valL,   featureCols.Length);

                    var model       = BuildPipeline(mlContext).Fit(trainView);
                    var predictions = model.Transform(valView);
                    var metrics     = mlContext.Regression.Evaluate(
                        predictions, labelColumnName: "Label");
                    sw.Stop();

                    Console.WriteLine(
                        $"  {foldName,-10} {metrics.RootMeanSquaredError,10:F6} " +
                        $"{metrics.MeanAbsoluteError,10:F6} {metrics.RSquared,8:F4} " +
                        $"{valL.Length,8}  ({sw.Elapsed.TotalSeconds:F1}s)");

                    PrintTopFeatures(model, featureCols, valF, valL);
                }
            }

            // Final model on full dataset
            Console.WriteLine("\nTraining final model on full dataset...");
            var sw2 = Stopwatch.StartNew();

            var (rawFullF, fullL) = LoadCsv(fullCsvPath, featureIdx, targetIdx);

            var fullImputer = new KnnImputer(k: 5, maxDistanceCols: 100);
            fullImputer.Fit(rawFullF);
            var fullF    = fullImputer.Transform(rawFullF);
            var fullView = ToDataView(mlContext, fullF, fullL, featureCols.Length);

            var finalModel = BuildPipeline(mlContext).Fit(fullView);
            sw2.Stop();

            var modelPath = Path.Combine(modelDir, $"{targetCol}.zip");
            mlContext.Model.Save(finalModel, fullView.Schema, modelPath);
            Console.WriteLine($"Saved → {modelPath}  ({sw2.Elapsed.TotalSeconds:F1}s)");

            var metaPath = Path.ChangeExtension(modelPath, ".features.json");
            File.WriteAllText(metaPath, JsonSerializer.Serialize(
                new ModelMeta { TargetColumn = targetCol, FeatureColumns = featureCols },
                new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"Saved → {metaPath}");
        }

        Console.WriteLine($"\n{new string('=', 60)}");
        Console.WriteLine("Training complete.");
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
