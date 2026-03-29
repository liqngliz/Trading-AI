namespace Predictor;

// Schema used for ML.NET DataView — Features vector + Label scalar
public sealed class ModelInput
{
    public float[] Features { get; set; } = [];
    public float   Label    { get; set; }
}
