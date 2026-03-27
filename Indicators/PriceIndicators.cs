using Integrations.TwelveData;

namespace Indicators;

// ── Result types for multi-value indicators ───────────────────────────────────

public readonly record struct MacdValue(decimal Line, decimal Signal, decimal Histogram);

public readonly record struct BollingerValue(
    decimal Upper,
    decimal Middle,
    decimal Lower,
    decimal PercentB,
    decimal Bandwidth);

public readonly record struct StochasticValue(decimal K, decimal D);

public readonly record struct AdxValue(decimal Adx, decimal PlusDI, decimal MinusDI);

// ── Indicator computations ────────────────────────────────────────────────────

public static class PriceIndicators
{
    // ── Helper ────────────────────────────────────────────────────────────────

    private static List<TimeSeriesValue> Ordered(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        bool skipFilled) =>
        series.Values
            .Where(v => !skipFilled || !v.IsFilled)
            .OrderBy(v => v.Datetime)
            .ToList();

    // ── From Close ────────────────────────────────────────────────────────────

    /// <summary>Natural log return: log(close[i] / close[i-1])</summary>
    public static Dictionary<DateTime, decimal> LogReturn(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        for (var i = 1; i < c.Count; i++)
        {
            var lr = Math.Log((double)(c[i].Close / c[i - 1].Close));
            if (double.IsFinite(lr))
                result[c[i].Datetime] = (decimal)lr;
        }
        return result;
    }

    /// <summary>Simple Moving Average of Close.</summary>
    public static Dictionary<DateTime, decimal> Sma(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        decimal sum = 0;
        for (var i = 0; i < c.Count; i++)
        {
            sum += c[i].Close;
            if (i >= period) sum -= c[i - period].Close;
            if (i >= period - 1) result[c[i].Datetime] = sum / period;
        }
        return result;
    }

    /// <summary>Exponential Moving Average of Close.</summary>
    public static Dictionary<DateTime, decimal> Ema(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        var k = 2m / (period + 1);
        decimal ema = 0, seedSum = 0;
        for (var i = 0; i < c.Count; i++)
        {
            if (i < period - 1) { seedSum += c[i].Close; continue; }
            if (i == period - 1) { ema = (seedSum + c[i].Close) / period; result[c[i].Datetime] = ema; continue; }
            ema += (c[i].Close - ema) * k;
            result[c[i].Datetime] = ema;
        }
        return result;
    }

    /// <summary>Relative Strength Index using Wilder smoothing.</summary>
    public static Dictionary<DateTime, decimal> Rsi(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period = 14,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        decimal avgGain = 0, avgLoss = 0;
        int count = 0;

        for (var i = 1; i < c.Count; i++)
        {
            var change = c[i].Close - c[i - 1].Close;
            var gain = change > 0 ? change : 0m;
            var loss = change < 0 ? -change : 0m;
            count++;

            if (count < period)
            {
                avgGain += gain;
                avgLoss += loss;
            }
            else if (count == period)
            {
                avgGain = (avgGain + gain) / period;
                avgLoss = (avgLoss + loss) / period;
                result[c[i].Datetime] = RsiValue(avgGain, avgLoss);
            }
            else
            {
                avgGain = (avgGain * (period - 1) + gain) / period;
                avgLoss = (avgLoss * (period - 1) + loss) / period;
                result[c[i].Datetime] = RsiValue(avgGain, avgLoss);
            }
        }
        return result;

        static decimal RsiValue(decimal gain, decimal loss) =>
            loss == 0 ? 100m : 100m - 100m / (1m + gain / loss);
    }

    /// <summary>Rate of Change: (close - close[period]) / close[period]</summary>
    public static Dictionary<DateTime, decimal> Roc(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period = 1,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        for (var i = period; i < c.Count; i++)
            result[c[i].Datetime] = (c[i].Close - c[i - period].Close) / c[i - period].Close;
        return result;
    }

    /// <summary>Rolling standard deviation of log returns.</summary>
    public static Dictionary<DateTime, decimal> RollingStdDev(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var returns = new decimal?[c.Count - 1];
        for (var i = 1; i < c.Count; i++)
        {
            var lr = Math.Log((double)(c[i].Close / c[i - 1].Close));
            returns[i - 1] = double.IsFinite(lr) ? (decimal)lr : null;
        }

        var result = new Dictionary<DateTime, decimal>(c.Count);
        for (var i = period - 1; i < returns.Length; i++)
        {
            decimal sum = 0, sumSq = 0;
            int valid = 0;
            for (var j = i - period + 1; j <= i; j++)
            {
                if (returns[j] is not { } r) continue;
                sum += r; sumSq += r * r; valid++;
            }
            if (valid < 2) continue;
            var mean = sum / valid;
            var variance = (double)(sumSq / valid - mean * mean);
            if (double.IsFinite(variance) && variance >= 0)
                result[c[i + 1].Datetime] = (decimal)Math.Sqrt(variance);
        }
        return result;
    }

    // ── From High, Low, Close ─────────────────────────────────────────────────

    /// <summary>Average True Range using Wilder smoothing.</summary>
    public static Dictionary<DateTime, decimal> Atr(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period = 14,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        decimal atr = 0, trSum = 0;
        int count = 0;

        for (var i = 1; i < c.Count; i++)
        {
            var tr = Math.Max(c[i].High - c[i].Low,
                     Math.Max(Math.Abs(c[i].High - c[i - 1].Close),
                              Math.Abs(c[i].Low  - c[i - 1].Close)));
            count++;
            if (count < period) { trSum += tr; }
            else if (count == period) { atr = (trSum + tr) / period; result[c[i].Datetime] = atr; }
            else { atr = (atr * (period - 1) + tr) / period; result[c[i].Datetime] = atr; }
        }
        return result;
    }

    /// <summary>Bollinger Bands: Middle = SMA, Upper/Lower = ±multiplier×stddev.</summary>
    public static Dictionary<DateTime, BollingerValue> BollingerBands(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period = 20,
        decimal multiplier = 2m,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, BollingerValue>(c.Count);
        for (var i = period - 1; i < c.Count; i++)
        {
            decimal sum = 0, sumSq = 0;
            for (var j = i - period + 1; j <= i; j++) { sum += c[j].Close; sumSq += c[j].Close * c[j].Close; }
            var middle = sum / period;
            var variance = (double)(sumSq / period - middle * middle);
            if (!double.IsFinite(variance) || variance < 0) continue;
            var stddev = (decimal)Math.Sqrt(variance);
            var upper = middle + multiplier * stddev;
            var lower = middle - multiplier * stddev;
            var percentB = stddev == 0 ? 0.5m : (c[i].Close - lower) / (upper - lower);
            var bandwidth = middle == 0 ? 0m : (upper - lower) / middle;
            result[c[i].Datetime] = new BollingerValue(upper, middle, lower, percentB, bandwidth);
        }
        return result;
    }

    /// <summary>Commodity Channel Index.</summary>
    public static Dictionary<DateTime, decimal> Cci(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period = 20,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        for (var i = period - 1; i < c.Count; i++)
        {
            decimal sumTypical = 0;
            for (var j = i - period + 1; j <= i; j++)
                sumTypical += (c[j].High + c[j].Low + c[j].Close) / 3m;
            var meanTypical = sumTypical / period;

            decimal meanDev = 0;
            for (var j = i - period + 1; j <= i; j++)
                meanDev += Math.Abs((c[j].High + c[j].Low + c[j].Close) / 3m - meanTypical);
            meanDev /= period;

            var typicalPrice = (c[i].High + c[i].Low + c[i].Close) / 3m;
            result[c[i].Datetime] = meanDev == 0 ? 0m : (typicalPrice - meanTypical) / (0.015m * meanDev);
        }
        return result;
    }

    /// <summary>Williams %R: 0 = overbought, -100 = oversold.</summary>
    public static Dictionary<DateTime, decimal> WilliamsR(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period = 14,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        for (var i = period - 1; i < c.Count; i++)
        {
            decimal highestHigh = c[i - period + 1].High, lowestLow = c[i - period + 1].Low;
            for (var j = i - period + 2; j <= i; j++)
            {
                if (c[j].High > highestHigh) highestHigh = c[j].High;
                if (c[j].Low  < lowestLow)  lowestLow  = c[j].Low;
            }
            var range = highestHigh - lowestLow;
            result[c[i].Datetime] = range == 0 ? 0m : -100m * (highestHigh - c[i].Close) / range;
        }
        return result;
    }

    /// <summary>Stochastic Oscillator: %K over kPeriod, %D = SMA(dPeriod) of %K.</summary>
    public static Dictionary<DateTime, StochasticValue> Stochastic(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int kPeriod = 14,
        int dPeriod = 3,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var kLine = new List<(DateTime Dt, decimal K)>(c.Count);

        for (var i = kPeriod - 1; i < c.Count; i++)
        {
            decimal highestHigh = c[i - kPeriod + 1].High, lowestLow = c[i - kPeriod + 1].Low;
            for (var j = i - kPeriod + 2; j <= i; j++)
            {
                if (c[j].High > highestHigh) highestHigh = c[j].High;
                if (c[j].Low  < lowestLow)  lowestLow  = c[j].Low;
            }
            var range = highestHigh - lowestLow;
            kLine.Add((c[i].Datetime, range == 0 ? 50m : 100m * (c[i].Close - lowestLow) / range));
        }

        var result = new Dictionary<DateTime, StochasticValue>(kLine.Count);
        for (var i = dPeriod - 1; i < kLine.Count; i++)
        {
            decimal dSum = 0;
            for (var j = i - dPeriod + 1; j <= i; j++) dSum += kLine[j].K;
            result[kLine[i].Dt] = new StochasticValue(kLine[i].K, dSum / dPeriod);
        }
        return result;
    }

    // ── From Open, High, Low, Close ───────────────────────────────────────────

    /// <summary>MACD: Line = EMA(fast)−EMA(slow), Signal = EMA(signalPeriod) of Line.</summary>
    public static Dictionary<DateTime, MacdValue> Macd(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int fastPeriod = 12,
        int slowPeriod = 26,
        int signalPeriod = 9,
        bool skipFilled = true)
    {
        var fastEma = Ema(series, fastPeriod, skipFilled);
        var slowEma = Ema(series, slowPeriod, skipFilled);

        var macdLine = slowEma.Keys
            .Where(fastEma.ContainsKey)
            .OrderBy(dt => dt)
            .Select(dt => (Dt: dt, Value: fastEma[dt] - slowEma[dt]))
            .ToList();

        var k = 2m / (signalPeriod + 1);
        decimal signal = 0, seedSum = 0;
        var result = new Dictionary<DateTime, MacdValue>(macdLine.Count);

        for (var i = 0; i < macdLine.Count; i++)
        {
            var (dt, macd) = macdLine[i];
            if (i < signalPeriod - 1) { seedSum += macd; continue; }
            if (i == signalPeriod - 1) signal = (seedSum + macd) / signalPeriod;
            else signal += (macd - signal) * k;
            result[dt] = new MacdValue(macd, signal, macd - signal);
        }
        return result;
    }

    /// <summary>(High − Low) / Close</summary>
    public static Dictionary<DateTime, decimal> CandleRange(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        foreach (var v in c)
            result[v.Datetime] = v.Close == 0 ? 0m : (v.High - v.Low) / v.Close;
        return result;
    }

    /// <summary>|Close − Open| / Close</summary>
    public static Dictionary<DateTime, decimal> BodySize(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        foreach (var v in c)
            result[v.Datetime] = v.Close == 0 ? 0m : Math.Abs(v.Close - v.Open) / v.Close;
        return result;
    }

    /// <summary>(High − max(Open, Close)) / Close</summary>
    public static Dictionary<DateTime, decimal> UpperWick(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        foreach (var v in c)
            result[v.Datetime] = v.Close == 0 ? 0m : (v.High - Math.Max(v.Open, v.Close)) / v.Close;
        return result;
    }

    /// <summary>(min(Open, Close) − Low) / Close</summary>
    public static Dictionary<DateTime, decimal> LowerWick(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        foreach (var v in c)
            result[v.Datetime] = v.Close == 0 ? 0m : (Math.Min(v.Open, v.Close) - v.Low) / v.Close;
        return result;
    }

    /// <summary>Close / rolling_max(High, period) − 1  (negative = below recent high)</summary>
    public static Dictionary<DateTime, decimal> DistanceFromHighN(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        for (var i = period - 1; i < c.Count; i++)
        {
            var high = c[i - period + 1].High;
            for (var j = i - period + 2; j <= i; j++)
                if (c[j].High > high) high = c[j].High;
            result[c[i].Datetime] = high == 0 ? 0m : c[i].Close / high - 1m;
        }
        return result;
    }

    /// <summary>
    /// Average Directional Index with +DI and -DI.
    /// Warmup = 2 × period bars. Values normalised to [0, 100].
    /// </summary>
    public static Dictionary<DateTime, AdxValue> Adx(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period = 14,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, AdxValue>(c.Count);

        decimal smTr = 0, smPlusDm = 0, smMinusDm = 0;
        decimal smAdx = 0, dxSum = 0;

        for (int i = 1; i < c.Count; i++)
        {
            var tr       = Math.Max(c[i].High - c[i].Low,
                           Math.Max(Math.Abs(c[i].High - c[i - 1].Close),
                                    Math.Abs(c[i].Low  - c[i - 1].Close)));
            var upMove   = c[i].High - c[i - 1].High;
            var downMove = c[i - 1].Low - c[i].Low;
            var plusDm   = upMove > 0 && upMove >= downMove ? upMove : 0m;
            var minusDm  = downMove > 0 && downMove > upMove ? downMove : 0m;

            if (i <= period)
            {
                // Seed phase: accumulate raw sums
                smTr += tr; smPlusDm += plusDm; smMinusDm += minusDm;
                if (i < period) continue;
                // i == period: first smoothed values are the sums; fall through to DI/DX
            }
            else
            {
                // Wilder smoothing: s = s − s/N + new
                smTr      = smTr      - smTr / period      + tr;
                smPlusDm  = smPlusDm  - smPlusDm / period  + plusDm;
                smMinusDm = smMinusDm - smMinusDm / period + minusDm;
            }

            if (smTr == 0) continue;
            var plusDI  = 100m * smPlusDm  / smTr;
            var minusDI = 100m * smMinusDm / smTr;
            var diSum   = plusDI + minusDI;
            var dx      = diSum == 0 ? 0m : 100m * Math.Abs(plusDI - minusDI) / diSum;

            // Seed ADX with average of first `period` DX values, then Wilder smooth
            int dxBar = i - period;   // 0-based index into DX series
            if (dxBar < period - 1)
            {
                dxSum += dx;
            }
            else if (dxBar == period - 1)
            {
                smAdx = (dxSum + dx) / period;
                result[c[i].Datetime] = new AdxValue(smAdx, plusDI, minusDI);
            }
            else
            {
                smAdx = (smAdx * (period - 1) + dx) / period;
                result[c[i].Datetime] = new AdxValue(smAdx, plusDI, minusDI);
            }
        }
        return result;
    }

    /// <summary>Close / rolling_min(Low, period) − 1  (positive = above recent low)</summary>
    public static Dictionary<DateTime, decimal> DistanceFromLowN(
        IReadOnlyDictionary<DateTime, TimeSeriesValue> series,
        int period,
        bool skipFilled = true)
    {
        var c = Ordered(series, skipFilled);
        var result = new Dictionary<DateTime, decimal>(c.Count);
        for (var i = period - 1; i < c.Count; i++)
        {
            var low = c[i - period + 1].Low;
            for (var j = i - period + 2; j <= i; j++)
                if (c[j].Low < low) low = c[j].Low;
            result[c[i].Datetime] = low == 0 ? 0m : c[i].Close / low - 1m;
        }
        return result;
    }
}
