namespace Importer;

/// <summary>
/// Describes one NaN-rate bucket: the CSV suffix, the upper NaN threshold,
/// the imputer to use, and a human-readable label.
/// Written to *_buckets.json by the Transformer; consumed by the Trainer and Predictor.
/// </summary>
public sealed class BucketDef
{
    public string Suffix    { get; set; } = "";
    public double MaxNanPct { get; set; }
    public string Imputer   { get; set; } = "knn";
    public string Label     { get; set; } = "";
}
