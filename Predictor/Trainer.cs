using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace Predictor;

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

        // ── Discover schema from CSV header ───────────────────────────────────

        var allCols = File.ReadLines(fullCsvPath).First().Split(',');

        // Features: every column except Timestamp and any _Target_ column
        var featureCols = allCols
            .Skip(1)
            .Where(c => !c.Contains("_Target_"))
            .ToArray();

        // Targets: _Return columns only (skip _Quintile — derived labels)
        var targetCols = allCols
            .Where(c => c.Contains("_Target_") && c.EndsWith("_Return"))
            .ToArray();

        Console.WriteLine($"Features : {featureCols.Length}");
        Console.WriteLine($"Targets  : {string.Join(", ", targetCols)}");

        // ── Build TextLoader ───────────────────────────────────────────────────

        var loaderColumns = new[] { new TextLoader.Column("Timestamp", DataKind.String, 0) }
            .Concat(allCols.Skip(1).Select((name, i) =>
                new TextLoader.Column(name, DataKind.Single, i + 1)))
            .ToArray();

        var loaderOptions = new TextLoader.Options
        {
            Columns            = loaderColumns,
            HasHeader          = true,
            Separators         = [','],
            MissingRealsAsNaNs = true,
        };

        var loader = mlContext.Data.CreateTextLoader(loaderOptions);

        // ── Discover walk-forward fold files ──────────────────────────────────

        var foldTrainFiles = Directory.GetFiles(datasetDir, "fold_*_train.csv")
            .OrderBy(f => f)
            .ToArray();

        Console.WriteLine($"Folds    : {foldTrainFiles.Length}");

        // ── Train one model per target horizon ────────────────────────────────

        foreach (var targetCol in targetCols)
        {
            Console.WriteLine($"\n{new string('=', 60)}");
            Console.WriteLine($"Target: {targetCol}");
            Console.WriteLine(new string('=', 60));

            var pipeline = BuildPipeline(mlContext, featureCols, targetCol);

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

                    var trainData = mlContext.Data.FilterRowsByMissingValues(
                        loader.Load(trainFile), targetCol);
                    var valData = mlContext.Data.FilterRowsByMissingValues(
                        loader.Load(valFile), targetCol);

                    var sw = Stopwatch.StartNew();
                    var model      = pipeline.Fit(trainData);
                    var predictions = model.Transform(valData);
                    var metrics    = mlContext.Regression.Evaluate(
                        predictions, labelColumnName: targetCol);
                    sw.Stop();

                    var rowCount = mlContext.Data.CreateEnumerable<object>(
                        valData, reuseRowObject: false).LongCount();

                    Console.WriteLine(
                        $"  {foldName,-10} {metrics.RootMeanSquaredError,10:F6} " +
                        $"{metrics.MeanAbsoluteError,10:F6} {metrics.RSquared,8:F4} " +
                        $"{rowCount,8}  ({sw.Elapsed.TotalSeconds:F1}s)");
                }
            }

            // Final model trained on the full dataset
            Console.WriteLine($"\nTraining final model on full dataset...");
            var sw2 = Stopwatch.StartNew();

            var fullData   = mlContext.Data.FilterRowsByMissingValues(
                loader.Load(fullCsvPath), targetCol);
            var finalModel = pipeline.Fit(fullData);
            sw2.Stop();

            var modelPath = Path.Combine(modelDir, $"{targetCol}.zip");
            mlContext.Model.Save(finalModel, fullData.Schema, modelPath);
            Console.WriteLine($"Saved → {modelPath}  ({sw2.Elapsed.TotalSeconds:F1}s)");
        }

        Console.WriteLine($"\n{new string('=', 60)}");
        Console.WriteLine("Training complete.");
    }

    private static IEstimator<ITransformer> BuildPipeline(
        MLContext mlContext,
        string[] featureCols,
        string labelCol)
    {
        return mlContext.Transforms
            .Concatenate("Features", featureCols)
            .Append(mlContext.Regression.Trainers.LightGbm(
                new LightGbmRegressionTrainer.Options
                {
                    LabelColumnName            = labelCol,
                    FeatureColumnName          = "Features",
                    NumberOfLeaves             = 63,
                    MinimumExampleCountPerLeaf = 20,
                    LearningRate               = 0.05,
                    NumberOfIterations         = 500,
                    Verbose                    = false,
                }));
    }
}
