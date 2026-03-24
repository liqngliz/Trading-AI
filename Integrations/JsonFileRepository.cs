using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace Integrations.TwelveData;

public sealed class JsonFileRepository<T> : IRepository<T>
{
    private readonly string _baseDirectory;
    private readonly JsonTypeInfo<T> _typeInfo;

    public JsonFileRepository(string baseDirectory, JsonTypeInfo<T> typeInfo)
    {
        if (string.IsNullOrWhiteSpace(baseDirectory))
            throw new ArgumentException("Base directory is required.", nameof(baseDirectory));

        _baseDirectory = baseDirectory;
        _typeInfo      = typeInfo;
        Directory.CreateDirectory(_baseDirectory);
    }

    public async Task<T?> GetAsync(string symbol, string interval, CancellationToken cancellationToken = default)
    {
        var path = GetPath(symbol, interval);

        if (!File.Exists(path))
            return default;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            var bytes  = await File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
            var readMs = sw.ElapsedMilliseconds;
            var result = JsonSerializer.Deserialize(bytes.AsSpan(), _typeInfo);
            Console.WriteLine($"[JsonFileRepository] GetAsync '{symbol}/{interval}': read={readMs}ms parse={sw.ElapsedMilliseconds - readMs}ms total={sw.ElapsedMilliseconds}ms ({bytes.Length / 1024}KB)");
            return result;
        }
        catch (JsonException ex)
        {
            Console.Error.WriteLine($"[JsonFileRepository] Corrupted cache file '{path}': {ex.Message}");
            Console.Error.WriteLine($"  Deleting corrupt file so it can be re-fetched.");
            File.Delete(path);
            return default;
        }
    }

    public async Task SaveAsync(string symbol, string interval, T value, CancellationToken cancellationToken = default)
    {
        var path = GetPath(symbol, interval);

        await using var stream = File.Create(path);
        await JsonSerializer.SerializeAsync(stream, value, _typeInfo, cancellationToken)
            .ConfigureAwait(false);
    }

    private string GetPath(string symbol, string interval) =>
        Path.Combine(_baseDirectory, $"{GetSafeKey(symbol)}_{GetSafeKey(interval)}.json");

    private static string GetSafeKey(string key) =>
        string.Concat(key.Select(c => Path.GetInvalidFileNameChars().Contains(c) ? '_' : c));
}
