namespace Predictor;

/// <summary>
/// Sidecar metadata saved alongside each trained model (.features.json).
/// Required at inference time to map dataset columns → the Features vector.
/// </summary>
public sealed class ModelMeta
{
    public string   TargetColumn   { get; set; } = "";
    public string[] FeatureColumns { get; set; } = [];
}
