namespace Imputers;

/// <summary>
/// Simple column-mean imputer.
/// For rows with &gt;50% NaN, KNN degenerates to mean imputation anyway
/// (shared column overlap is too small for meaningful distances) but is
/// orders of magnitude slower.  Use this imputer for the 50–75% and 75–100%
/// NaN-rate buckets.
/// </summary>
public sealed class MeanImputer
{
    private float[] _colMeans = [];

    public void Fit(float[][] data)
    {
        if (data.Length == 0) return;
        int nCols  = data[0].Length;
        var sums   = new double[nCols];
        var counts = new int[nCols];

        foreach (var row in data)
            for (int c = 0; c < nCols; c++)
                if (!float.IsNaN(row[c])) { sums[c] += row[c]; counts[c]++; }

        _colMeans = new float[nCols];
        for (int c = 0; c < nCols; c++)
            _colMeans[c] = counts[c] > 0 ? (float)(sums[c] / counts[c]) : 0f;
    }

    public float[][] Transform(float[][] data)
    {
        if (_colMeans.Length == 0)
            throw new InvalidOperationException("Call Fit before Transform.");

        int nCols  = data[0].Length;
        var result = new float[data.Length][];
        for (int i = 0; i < data.Length; i++)
        {
            result[i] = (float[])data[i].Clone();
            for (int c = 0; c < nCols; c++)
                if (float.IsNaN(result[i][c]))
                    result[i][c] = c < _colMeans.Length ? _colMeans[c] : 0f;
        }
        return result;
    }
}
