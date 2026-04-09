using Predictor;

// Usage:
//   dotnet run -- --train   [--dataset /path/to/Dataset] [--folds N] [--holdout-days N]
//   dotnet run -- --predict [--dataset /path/to/Dataset]
//
// Scans every xauusd_* subfolder under the dataset root.
// --train        : trains models and saves them to <subfolder>/Models/
// --predict      : loads trained models and prints predictions for the latest bar
// --holdout-days : holdout size as calendar days (default: 90)
//                  e.g. --holdout-days 180  → last 180 days of data held out

var mode = args.Contains("--train") ? "train"
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

// Holdout % per symbol/timeframe directory name prefix.
// Key is matched as a case-insensitive prefix of the directory name (e.g. "xauusd_4h").
// First match wins. Falls back to DefaultHoldoutPct if nothing matches.
const double DefaultHoldoutPct = 0.15;
var HoldoutConfig = new (string Prefix, double Pct)[]
{
    ("xauusd_1h",   0.15),
    ("xauusd_1day", 0.15),
    ("xauusd_1week",0.15),
    ("xagusd_1h",   0.15),
    ("xagusd_1day", 0.15),
    ("xagusd_1week",0.15),
    ("spy_1h",      0.15),
    ("spy_1day",    0.15),
    ("qqq_1h",      0.15),
    ("qqq_1day",    0.15),
};

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

if (mode == "train")
{
    var numFolds    = int.TryParse(GetArg(args, "--folds"),        out var nf) ? nf : 6;
    var holdoutDays = int.TryParse(GetArg(args, "--holdout-days"), out var hd) ? (int?)hd : 90;

    Console.WriteLine($"CV folds     : {numFolds}");
    if (holdoutDays.HasValue)
        Console.WriteLine($"Holdout      : {holdoutDays.Value} days (overrides per-symbol %)");

    foreach (var horizonDir in horizonDirs)
    {
        var modelDir = Path.Combine(horizonDir, "Models");
        Console.WriteLine($"\n{new string('=', 60)}");
        Console.WriteLine($"Horizon : {Path.GetFileName(horizonDir)}");
        Console.WriteLine($"Dataset : {horizonDir}");
        Console.WriteLine($"Models  : {modelDir}");
        Directory.CreateDirectory(modelDir);
        var dirName    = Path.GetFileName(horizonDir);
        var holdoutPct = HoldoutConfig.FirstOrDefault(h => dirName.StartsWith(h.Prefix, StringComparison.OrdinalIgnoreCase)).Pct;
        if (holdoutPct == 0) holdoutPct = DefaultHoldoutPct;
        ModelTrainer.Run(horizonDir, modelDir, numFolds, holdoutPct, purge: 1, embargo: 50, holdoutDays: holdoutDays);
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
