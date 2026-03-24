namespace Integrations.TwelveData;

internal sealed class TwelveDataRateLimitException(string message) : Exception(message);

/// <summary>
/// Proactively throttles API calls to stay within the TwelveData per-minute credit limit,
/// and exposes the delay to use when a 429 response is received.
/// </summary>
internal static class TwelveDataRateLimiter
{
    private static readonly SemaphoreSlim Lock = new(1, 1);
    private static readonly Queue<DateTimeOffset> RequestTimes = new();
    private const int MaxPerMinute = 55;
    private static readonly TimeSpan Window = TimeSpan.FromSeconds(60);

    /// <summary>How long to wait after receiving a 429 before retrying (just over one minute).</summary>
    public static readonly TimeSpan RetryDelay = TimeSpan.FromSeconds(61);

    /// <summary>
    /// Waits until a request slot is available within the rate limit window,
    /// then records the current request.
    /// </summary>
    public static async Task WaitForSlotAsync(CancellationToken ct = default)
    {
        await Lock.WaitAsync(ct);
        bool held = true;
        try
        {
            Trim();

            if (RequestTimes.Count >= MaxPerMinute)
            {
                var delay = RequestTimes.Peek() + Window - DateTimeOffset.UtcNow;
                if (delay > TimeSpan.Zero)
                {
                    Console.WriteLine($"  [Rate limit] {RequestTimes.Count}/{MaxPerMinute} req/min reached — waiting {delay.TotalSeconds:F0}s...");
                    Lock.Release();
                    held = false;
                    await Task.Delay(delay, ct);
                    await Lock.WaitAsync(ct);
                    held = true;
                    Trim();
                }
            }

            RequestTimes.Enqueue(DateTimeOffset.UtcNow);
        }
        finally
        {
            if (held) Lock.Release();
        }
    }

    private static void Trim()
    {
        var cutoff = DateTimeOffset.UtcNow - Window;
        while (RequestTimes.Count > 0 && RequestTimes.Peek() <= cutoff)
            RequestTimes.Dequeue();
    }
}
