namespace Predictor;

/// <summary>
/// K-nearest-neighbour multivariate imputer.
/// Fit on training data; each NaN cell is filled with the mean of the k
/// closest reference rows. Distance is computed on the most-complete columns
/// (z-scored, partial distance when either row has NaN in a column).
/// Falls back to the column mean when no neighbour has a value for a cell.
/// </summary>
public sealed class KnnImputer
{
    private readonly int _k;
    private readonly int _maxDistanceCols;

    private float[][]? _refData;
    private float[]    _colMeans     = [];
    private int[]      _distanceCols = [];
    private float[]    _dcStds       = [];

    public KnnImputer(int k = 5, int maxDistanceCols = 100)
    {
        _k = k;
        _maxDistanceCols = maxDistanceCols;
    }

    /// <summary>
    /// Stores reference data and computes column statistics and distance columns.
    /// </summary>
    public void Fit(float[][] data)
    {
        if (data.Length == 0) return;
        int nCols = data[0].Length;
        _refData = data;

        // Column means and NaN counts
        var sums      = new double[nCols];
        var counts    = new int[nCols];
        var nanCounts = new int[nCols];

        foreach (var row in data)
        {
            for (int c = 0; c < nCols; c++)
            {
                if (!float.IsNaN(row[c])) { sums[c] += row[c]; counts[c]++; }
                else nanCounts[c]++;
            }
        }

        _colMeans = new float[nCols];
        for (int c = 0; c < nCols; c++)
            _colMeans[c] = counts[c] > 0 ? (float)(sums[c] / counts[c]) : 0f;

        // Distance columns: most-complete (fewest NaN)
        _distanceCols = Enumerable.Range(0, nCols)
            .OrderBy(c => nanCounts[c])
            .Take(Math.Min(_maxDistanceCols, nCols))
            .ToArray();

        // Std-dev of each distance column (for z-scoring)
        _dcStds = new float[_distanceCols.Length];
        for (int di = 0; di < _distanceCols.Length; di++)
        {
            int c = _distanceCols[di];
            double var2 = 0; int cnt = 0;
            foreach (var row in data)
            {
                if (!float.IsNaN(row[c]))
                {
                    double d = row[c] - _colMeans[c];
                    var2 += d * d; cnt++;
                }
            }
            _dcStds[di] = cnt > 1 ? (float)Math.Sqrt(var2 / cnt) : 1f;
        }
    }

    /// <summary>
    /// Returns a new float[][] with every NaN replaced by the mean of the
    /// k nearest reference rows. Rows without any NaN are cloned unchanged.
    /// </summary>
    public float[][] Transform(float[][] data)
    {
        if (_refData is null)
            throw new InvalidOperationException("Call Fit before Transform.");

        int nCols = data[0].Length;
        int nRef  = _refData.Length;
        var result    = new float[data.Length][];
        var distances = new float[nRef];

        for (int i = 0; i < data.Length; i++)
        {
            result[i] = (float[])data[i].Clone();
            var row = data[i];

            bool hasMissing = false;
            for (int c = 0; c < nCols; c++)
                if (float.IsNaN(row[c])) { hasMissing = true; break; }
            if (!hasMissing) continue;

            // Compute partial z-scored Euclidean distance to each reference row
            for (int r = 0; r < nRef; r++)
            {
                var refRow = _refData[r];
                float dist = 0f; int shared = 0;
                for (int di = 0; di < _distanceCols.Length; di++)
                {
                    int   c   = _distanceCols[di];
                    float a   = row[c], b = refRow[c];
                    if (!float.IsNaN(a) && !float.IsNaN(b))
                    {
                        float std  = _dcStds[di] < 1e-8f ? 1f : _dcStds[di];
                        float diff = (a - b) / std;
                        dist  += diff * diff;
                        shared++;
                    }
                }
                distances[r] = shared > 0 ? dist / shared : float.MaxValue;
            }

            // k nearest neighbour indices
            var neighbours = distances
                .Select((d, idx) => (d, idx))
                .OrderBy(x => x.d)
                .Take(_k)
                .Select(x => x.idx)
                .ToArray();

            // Fill each NaN cell with the mean of neighbour values
            for (int c = 0; c < nCols; c++)
            {
                if (!float.IsNaN(row[c])) continue;
                double sum = 0; int cnt = 0;
                foreach (int ni in neighbours)
                {
                    float v = _refData[ni][c];
                    if (!float.IsNaN(v)) { sum += v; cnt++; }
                }
                result[i][c] = cnt > 0 ? (float)(sum / cnt) : _colMeans[c];
            }
        }

        return result;
    }
}
