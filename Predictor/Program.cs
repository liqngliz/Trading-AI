using Predictor;

// Usage:
//   dotnet run -- --train   [--dataset /path/to/Dataset] [--folds N]
//   dotnet run -- --retrain [--dataset /path/to/Dataset] [--folds N]
//   dotnet run -- --predict [--dataset /path/to/Dataset]
//
// Scans every xauusd_* subfolder under the dataset root.
// --train  : shuffles bucket CSVs, saves *_shuffled.csv, trains models
// --retrain: reuses existing *_shuffled.csv (no re-shuffle); same as --train on first run
// --predict: loads trained models and prints predictions for the latest bar

var mode = args.Contains("--train")   ? "train"
         : args.Contains("--retrain") ? "retrain"
         : args.Contains("--predict") ? "predict"
         : "train"; // default to train if no mode specified

if (mode is null)
{
    Console.WriteLine("ERROR: Specify --train or --predict.");
    return 1;
}

var datasetRoot = GetArg(args, "--dataset")
    ?? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Dataset"));

Console.WriteLine($"Mode         : {mode}");
Console.WriteLine($"Dataset root : {datasetRoot}");

if (!Directory.Exists(datasetRoot))
{
    Console.WriteLine($"ERROR: Dataset root not found: {datasetRoot}");
    Console.WriteLine("Run the Importer first, or pass --dataset <path>.");
    return 1;
}

string[] horizonPatterns = ["xauusd_*", "xagusd_*", "spy_*", "qqq_*"];
var horizonDirs = horizonPatterns
    .SelectMany(p => Directory.GetDirectories(datasetRoot, p))
    .OrderBy(d => d)
    .ToArray();

if (horizonDirs.Length == 0)
{
    Console.WriteLine("ERROR: No subdirectories found. Run the Importer first.");
    return 1;
}

if (mode is "train" or "retrain")
{
    bool retrain = mode == "retrain";
    var numFolds = int.TryParse(GetArg(args, "--folds"), out var nf) ? nf : 6;
    Console.WriteLine($"CV folds     : {numFolds}");
    Console.WriteLine($"Retrain mode : {(retrain ? "yes (reuse existing shuffled CSVs)" : "no (re-shuffle)")}");

    foreach (var horizonDir in horizonDirs)
    {
        var modelDir = Path.Combine(horizonDir, "Models");
        Console.WriteLine($"\n{new string('=', 60)}");
        Console.WriteLine($"Horizon : {Path.GetFileName(horizonDir)}");
        Console.WriteLine($"Dataset : {horizonDir}");
        Console.WriteLine($"Models  : {modelDir}");
        Directory.CreateDirectory(modelDir);
        ModelTrainer.Run(horizonDir, modelDir, numFolds, retrain);
    }
}
else
{
    var predictDirs = horizonDirs
        .Where(d => Directory.Exists(Path.Combine(d, "Models")))
        .ToArray();

    if (predictDirs.Length == 0)
    {
        Console.WriteLine("ERROR: No xauusd_*/Models directories found. Run --train first.");
        return 1;
    }

    foreach (var horizonDir in predictDirs)
    {
        var modelDir = Path.Combine(horizonDir, "Models");
        Console.WriteLine($"\n{new string('=', 60)}");
        Console.WriteLine($"Horizon : {Path.GetFileName(horizonDir)}");
        PredictRunner.Run(horizonDir, modelDir);
    }
}

return 0;

static string? GetArg(string[] args, string name)
{
    var idx = Array.IndexOf(args, name);
    return idx >= 0 && idx + 1 < args.Length ? args[idx + 1] : null;
}
