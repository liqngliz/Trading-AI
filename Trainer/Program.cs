using Trainer;

// Usage:
//   dotnet run -- [--dataset /path/to/Dataset] [--folds N]
//
// Trains one model set per xauusd_* subfolder found under the dataset root.
// Models are saved to <subfolder>/Models/.

var datasetRoot = GetArg(args, "--dataset")
    ?? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Dataset"));

var numFolds = int.TryParse(GetArg(args, "--folds"), out var nf) ? nf : 10;

Console.WriteLine($"Dataset root : {datasetRoot}");
Console.WriteLine($"CV folds     : {numFolds}");

if (!Directory.Exists(datasetRoot))
{
    Console.WriteLine($"ERROR: Dataset root not found: {datasetRoot}");
    Console.WriteLine("Run the Importer first, or pass --dataset <path>.");
    return 1;
}

var horizonDirs = Directory.GetDirectories(datasetRoot, "xauusd_*")
    .OrderBy(d => d)
    .ToArray();

if (horizonDirs.Length == 0)
{
    Console.WriteLine("ERROR: No xauusd_* subdirectories found. Run the Importer first.");
    return 1;
}

foreach (var horizonDir in horizonDirs)
{
    var modelDir = Path.Combine(horizonDir, "Models");
    Console.WriteLine($"\n{new string('=', 60)}");
    Console.WriteLine($"Horizon : {Path.GetFileName(horizonDir)}");
    Console.WriteLine($"Dataset : {horizonDir}");
    Console.WriteLine($"Models  : {modelDir}");
    Directory.CreateDirectory(modelDir);
    ModelTrainer.Run(horizonDir, modelDir, numFolds);
}

return 0;

static string? GetArg(string[] args, string name)
{
    var idx = Array.IndexOf(args, name);
    return idx >= 0 && idx + 1 < args.Length ? args[idx + 1] : null;
}
