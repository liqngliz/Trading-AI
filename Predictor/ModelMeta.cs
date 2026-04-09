namespace Predictor;

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
    /// <summary>Per-feature mean of the winning CV training set (post-imputation). Used for drift detection.</summary>
    public double[] FeatureMean    { get; set; } = [];
    /// <summary>Per-feature std-dev of the winning CV training set (post-imputation). Used for drift detection.</summary>
    public double[] FeatureStd     { get; set; } = [];
    /// <summary>Per-feature normalised gain from the final production model (max = 1.0). Used for importance-weighted drift.</summary>
    public double[] FeatureGain    { get; set; } = [];
}
