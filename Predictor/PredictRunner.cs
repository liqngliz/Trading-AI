using System.Globalization;
using System.Text.Json;
using Imputers;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Predictor;

public static class PredictRunner
{
    // Must match the buckets produced by the Transformer and trained by the Trainer.
    // Ordered from best data quality (lowest NaN) to worst.
    private static readonly (string Suffix, double MaxNanPct, string Label)[] Buckets =
    [
        ("nan_0_66",   0.66, "≤66% NaN"),
        ("nan_66_100", 1.00, ">66% NaN"),
    ];

    /// <summary>
    /// For each target column, selects the appropriate NaN-bucket model based on
    /// the feature availability of the most recent dataset row, and prints the prediction.
    /// Uses KNN imputation for ≤50% NaN buckets and mean imputation for sparser buckets.
    /// </summary>
    public static void Run(string datasetDir, string modelDir)
    {
        var csvPath = Directory.GetFiles(datasetDir, "*_dataset.csv").FirstOrDefault();
        if (csvPath is null)
        {
            Console.WriteLine($"ERROR: No *_dataset.csv found in: {datasetDir}");
            return;
        }

        // ── Read header ───────────────────────────────────────────────────────
        var allCols  = File.ReadLines(csvPath).First().Split(',');
        var colIndex = allCols.Select((n, i) => (n, i)).ToDictionary(x => x.n, x => x.i);

        // Feature columns (used to compute NaN rate of the query row)
        var featureCols = allCols
            .Where(c => c != "Timestamp" && !c.Contains("_Target_"))
            .ToArray();

        // ── Load last 1000 rows (reference for imputation + query row) ────────
        const int refWindow = 1_000;
        var recentLines = File.ReadLines(csvPath)
            .Skip(1)
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .TakeLast(refWindow)
            .ToArray();

        if (recentLines.Length == 0)
        {
            Console.WriteLine("ERROR: Dataset is empty.");
            return;
        }

        var lastParts     = recentLines[^1].Split(',');
        var barTimestamp  = lastParts.Length > 0 ? lastParts[0] : "?";

        // ── Compute NaN rate of the last row (all feature columns) ────────────
        int nanCount = featureCols.Count(c =>
        {
            if (!colIndex.TryGetValue(c, out var idx)) return true;
            return idx >= lastParts.Length || string.IsNullOrEmpty(lastParts[idx]);
        });
        double nanPct = featureCols.Length > 0 ? (double)nanCount / featureCols.Length : 1.0;

        Console.WriteLine($"Bar           : {barTimestamp}");
        Console.WriteLine($"Feature NaN   : {nanCount}/{featureCols.Length}  ({nanPct:P1})");

        // Select the tightest bucket that covers the query row's NaN rate
        var bucket = Buckets.First(b => nanPct <= b.MaxNanPct);
        Console.WriteLine($"Model bucket  : {bucket.Label} ({bucket.Suffix})");
        Console.WriteLine();

        // ── Read TargetVolScalar from last row (used to de-scale prediction) ────
        // Key: targetCol name → vol scalar value (null if anomalous / warm-up not met)
        var volScalars = new Dictionary<string, double?>(StringComparer.Ordinal);
        foreach (var col in allCols.Where(c => c.Contains("_Target_") && c.EndsWith("_Return")))
        {
            // Corresponding TargetVolScalar column: strip "_Target_<tf>_Return" → "_TargetVolScalar"
            var prefix = col[..col.IndexOf("_Target_", StringComparison.Ordinal)];
            var vsCol  = prefix + "_TargetVolScalar";
            double? vs = null;
            if (colIndex.TryGetValue(vsCol, out var vsIdx)
                && vsIdx < lastParts.Length
                && !string.IsNullOrEmpty(lastParts[vsIdx])
                && double.TryParse(lastParts[vsIdx], NumberStyles.Float, CultureInfo.InvariantCulture, out var vsVal)
                && vsVal > 1e-5)
            {
                vs = vsVal;
            }
            volScalars[col] = vs;
        }

        // ── Find all models for this bucket ───────────────────────────────────
        var modelFiles = Directory.GetFiles(modelDir, $"*_{bucket.Suffix}.zip")
            .OrderBy(f => f)
            .ToArray();

        if (modelFiles.Length == 0)
        {
            Console.WriteLine($"No trained models found for bucket '{bucket.Suffix}' in: {modelDir}");
            Console.WriteLine("Run the Trainer first.");
            return;
        }

        var mlContext = new MLContext(seed: 42);

        foreach (var modelPath in modelFiles)
        {
            var metaPath = Path.ChangeExtension(modelPath, ".features.json");
            if (!File.Exists(metaPath))
            {
                Console.WriteLine($"[SKIP] {Path.GetFileName(modelPath)} — no .features.json sidecar");
                continue;
            }

            var meta = JsonSerializer.Deserialize<ModelMeta>(File.ReadAllText(metaPath))!;

            // Resolve model-specific feature indices
            var featureIndices = meta.FeatureColumns
                .Select(c => colIndex.TryGetValue(c, out var idx) ? idx : -1)
                .ToArray();

            var missingCols = meta.FeatureColumns.Where(c => !colIndex.ContainsKey(c)).ToArray();
            if (missingCols.Length > 0)
            {
                Console.WriteLine($"[SKIP] {meta.TargetColumn}/{bucket.Suffix} — {missingCols.Length} feature(s) not in current CSV (re-import)");
                continue;
            }

            // ── Parse feature rows ────────────────────────────────────────────
            var allFeatureRows = ParseFeatureRows(recentLines, featureIndices);
            var refData        = allFeatureRows.Length > 1 ? allFeatureRows[..^1] : allFeatureRows;
            var queryRow       = new[] { allFeatureRows[^1] };

            // ── Impute (use same imputer as at training time) ─────────────────
            float[][] imputedQuery;
            if (meta.Imputer == "knn")
            {
                var imputer = new KnnImputer(k: 5, maxDistanceCols: 100);
                imputer.Fit(refData);
                imputedQuery = imputer.Transform(queryRow);
            }
            else
            {
                var imputer = new MeanImputer();
                imputer.Fit(refData);
                imputedQuery = imputer.Transform(queryRow);
            }

            // ── Build single-row IDataView ────────────────────────────────────
            var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
            schemaDef["Features"].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, meta.FeatureColumns.Length);

            var dataView = mlContext.Data.LoadFromEnumerable(
                [new ModelInput { Features = imputedQuery[0], Label = 0f }],
                schemaDef);

            // ── Score ─────────────────────────────────────────────────────────
            var model     = mlContext.Model.Load(modelPath, out _);
            var predicted = model.Transform(dataView);
            var score     = predicted.GetColumn<float>("Score").First();

            volScalars.TryGetValue(meta.TargetColumn, out var curVol);
            bool anomalous = curVol is null;

            Console.WriteLine($"┌─ {meta.TargetColumn}  [{bucket.Label}]");
            Console.WriteLine($"│  Score  : {score:+0.0000;-0.0000}  (vol-normalised log return)");
            if (!anomalous)
            {
                double expectedReturn = score * curVol!.Value;
                Console.WriteLine($"│  Return : {expectedReturn:+0.0000%;-0.0000%}  (score × vol {curVol.Value:F5})");
            }
            else
            {
                Console.WriteLine($"│  Return : [UNRELIABLE — vol scalar missing or near-zero on this bar]");
            }
            Console.WriteLine($"│  Signal : {(score > 0 ? "LONG  ▲" : "SHORT ▼")}{(anomalous ? "  ⚠ anomalous bar" : "")}");
            Console.WriteLine($"└─ Model  : {Path.GetFileName(modelPath)}");
            Console.WriteLine();
        }
    }

    private static float[][] ParseFeatureRows(string[] lines, int[] featureIndices)
    {
        var rows = new float[lines.Length][];
        for (int i = 0; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',');
            var row   = new float[featureIndices.Length];
            for (int j = 0; j < featureIndices.Length; j++)
            {
                int fi  = featureIndices[j];
                row[j] = fi >= 0
                    && fi < parts.Length
                    && !string.IsNullOrEmpty(parts[fi])
                    && float.TryParse(parts[fi], NumberStyles.Float,
                           CultureInfo.InvariantCulture, out var fv)
                    ? fv
                    : float.NaN;
            }
            rows[i] = row;
        }
        return rows;
    }
}
