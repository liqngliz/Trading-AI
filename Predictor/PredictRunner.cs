using System.Globalization;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Predictor;

public static class PredictRunner
{
    /// <summary>
    /// For each trained model (.zip + .features.json) found in <paramref name="modelDir"/>,
    /// predicts on the most recent row of the dataset CSV and prints the result.
    /// </summary>
    public static void Run(string datasetDir, string modelDir)
    {
        var csvPath = Path.Combine(datasetDir, "xauusd_4h_dataset.csv");
        if (!File.Exists(csvPath))
        {
            Console.WriteLine($"ERROR: Dataset not found: {csvPath}");
            return;
        }

        var modelFiles = Directory.GetFiles(modelDir, "*.zip")
            .OrderBy(f => f)
            .ToArray();

        if (modelFiles.Length == 0)
        {
            Console.WriteLine($"No trained models found in: {modelDir}");
            Console.WriteLine("Run training first (omit --predict).");
            return;
        }

        // ── Read CSV header ───────────────────────────────────────────────────
        var allCols  = File.ReadLines(csvPath).First().Split(',');
        var colIndex = allCols.Select((n, i) => (n, i)).ToDictionary(x => x.n, x => x.i);

        // ── Load a recent window of rows as reference data for the imputer ────
        // Using the last 1 000 rows is far more than enough for KNN (k=5).
        const int refWindowSize = 1_000;
        var recentLines = File.ReadLines(csvPath)
            .Skip(1)
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .TakeLast(refWindowSize)
            .ToArray();

        if (recentLines.Length == 0)
        {
            Console.WriteLine("ERROR: Dataset is empty.");
            return;
        }

        // The very last bar in the CSV is the "current" bar:
        // its target column is NaN (the future close hasn't happened yet).
        // That is the row we predict on.
        var lastParts = recentLines[^1].Split(',');
        var barTimestamp = lastParts.Length > 0 ? lastParts[0] : "?";

        Console.WriteLine($"Predicting for bar : {barTimestamp}");
        Console.WriteLine($"Models directory   : {modelDir}");
        Console.WriteLine($"Reference rows     : {recentLines.Length - 1} (last {refWindowSize} minus query)");
        Console.WriteLine();

        var mlContext = new MLContext(seed: 42);

        foreach (var modelPath in modelFiles)
        {
            var metaPath = Path.ChangeExtension(modelPath, ".features.json");
            if (!File.Exists(metaPath))
            {
                Console.WriteLine($"[SKIP] {Path.GetFileName(modelPath)} — no .features.json sidecar (re-train to generate it)");
                continue;
            }

            var meta = JsonSerializer.Deserialize<ModelMeta>(File.ReadAllText(metaPath))!;

            // Resolve feature column indices
            var featureIndices = meta.FeatureColumns
                .Select(c => colIndex.TryGetValue(c, out var idx) ? idx : -1)
                .ToArray();

            var missingCols = meta.FeatureColumns
                .Where(c => !colIndex.ContainsKey(c))
                .ToArray();

            if (missingCols.Length > 0)
            {
                Console.WriteLine($"[SKIP] {meta.TargetColumn} — {missingCols.Length} feature(s) absent from current CSV (re-train or re-import)");
                continue;
            }

            // ── Parse feature rows ────────────────────────────────────────────
            var allFeatureRows = ParseFeatureRows(recentLines, featureIndices);

            // Reference = all rows except the last; query = last row
            var refData   = allFeatureRows.Length > 1
                ? allFeatureRows[..^1]
                : allFeatureRows;
            var queryRow  = new[] { allFeatureRows[^1] };

            // ── Impute ────────────────────────────────────────────────────────
            var imputer = new KnnImputer(k: 5, maxDistanceCols: 100);
            imputer.Fit(refData);
            var imputedQuery = imputer.Transform(queryRow);

            // ── Build single-row IDataView ────────────────────────────────────
            var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
            schemaDef["Features"].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, meta.FeatureColumns.Length);

            var dataView = mlContext.Data.LoadFromEnumerable(
                [new ModelInput { Features = imputedQuery[0], Label = 0f }],
                schemaDef);

            // ── Load model and score ──────────────────────────────────────────
            var model     = mlContext.Model.Load(modelPath, out _);
            var predicted = model.Transform(dataView);
            var score     = predicted.GetColumn<float>("Score").First();

            // ── Output ───────────────────────────────────────────────────────
            Console.WriteLine($"┌─ {meta.TargetColumn}");
            Console.WriteLine($"│  Score  : {score:+0.0000;-0.0000}  (vol-normalised log return)");
            Console.WriteLine($"│  Signal : {(score > 0 ? "LONG  ▲" : "SHORT ▼")}");
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
                row[j] = fi < parts.Length
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
