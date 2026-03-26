using System.Diagnostics;
using System.Globalization;
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
            .OrderBy(f => f).ToArray();

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
                Console.WriteLine($"  {"Fold",-10} {"RMSE",10} {"MAE",10} {"R²",8} {"Rows",8}");
                Console.WriteLine($"  {new string('-', 50)}");

                foreach (var trainFile in foldTrainFiles)
                {
                    var valFile = trainFile.Replace("_train.csv", "_val.csv");
                    if (!File.Exists(valFile)) continue;

                    var foldName = Path.GetFileNameWithoutExtension(trainFile)
                        .Replace("_train", "");

                    var (rawTrainF, trainL) = LoadCsv(trainFile, featureIdx, targetIdx);
                    var (rawValF,   valL)   = LoadCsv(valFile,   featureIdx, targetIdx);
                    if (trainL.Length == 0 || valL.Length == 0) continue;

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
    private static (float[][] Features, float[] Labels) LoadCsv(
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
