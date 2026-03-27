using Predictor;

// Usage:
//   Train:   dotnet run -- [--dataset /path/to/Dataset] [--models /path/to/Models]
//   Predict: dotnet run -- --predict [--dataset /path/to/Dataset] [--models /path/to/Models]

var datasetDir = GetArg(args, "--dataset")
    ?? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Dataset", "Training"));

var modelDir = GetArg(args, "--models")
    ?? Path.Combine(datasetDir, "Models");

var predictMode = args.Contains("--predict");

Console.WriteLine($"Mode    : {(predictMode ? "Predict" : "Train")}");
Console.WriteLine($"Dataset : {datasetDir}");
Console.WriteLine($"Models  : {modelDir}");
Console.WriteLine();

if (!Directory.Exists(datasetDir))
{
    Console.WriteLine($"ERROR: Dataset directory not found: {datasetDir}");
    Console.WriteLine("Run the Importer first, or pass --dataset <path>.");
    return 1;
}

if (predictMode)
{
    PredictRunner.Run(datasetDir, modelDir);
}
else
{
    Directory.CreateDirectory(modelDir);
    Trainer.Run(datasetDir, modelDir);
}

return 0;

static string? GetArg(string[] args, string name)
{
    var idx = Array.IndexOf(args, name);
    return idx >= 0 && idx + 1 < args.Length ? args[idx + 1] : null;
}
