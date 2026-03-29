namespace Trainer;

/// <summary>
/// Sidecar metadata saved alongside each trained model (.features.json).
/// Required at inference time to map dataset columns → the Features vector.
/// </summary>
public sealed class ModelMeta
{
    public string   TargetColumn   { get; set; } = "";
    /// <summary>NaN-rate bucket this model was trained on (e.g. "nan_0_33").</summary>
    public string   NanBucket      { get; set; } = "";
    /// <summary>Imputer used at training time: "knn" or "mean".</summary>
    public string   Imputer        { get; set; } = "knn";
    public string[] FeatureColumns { get; set; } = [];
}
