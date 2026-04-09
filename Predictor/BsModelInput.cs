namespace Predictor;

// Schema for the black-swan binary classifier (label = true when row is near a BS event)
public sealed class BsModelInput
{
    public float[] Features { get; set; } = [];
    public bool    Label    { get; set; }
}
