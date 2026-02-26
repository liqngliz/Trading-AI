using System.Text.Json;
using System.Text.Json.Serialization;

namespace Integrations.TwelveData;

public sealed class JsonFileRepository<T> : IRepository<T>
{
    private readonly string _baseDirectory;
    private readonly JsonSerializerOptions _jsonOptions;

    public JsonFileRepository(string baseDirectory)
    {
        if (string.IsNullOrWhiteSpace(baseDirectory))
            throw new ArgumentException("Base directory is required.", nameof(baseDirectory));

        _baseDirectory = baseDirectory;
        Directory.CreateDirectory(_baseDirectory);

        _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.Never
        };
    }

    public async Task<T?> GetAsync(string key, CancellationToken cancellationToken = default)
    {
        var path = GetPath(key);

        if (!File.Exists(path))
            return default;

        await using var stream = File.OpenRead(path);
        return await JsonSerializer.DeserializeAsync<T>(stream, _jsonOptions, cancellationToken)
            .ConfigureAwait(false);
    }

    public async Task SaveAsync(string key, T value, CancellationToken cancellationToken = default)
    {
        var path = GetPath(key);

        await using var stream = File.Create(path);
        await JsonSerializer.SerializeAsync(stream, value, _jsonOptions, cancellationToken)
            .ConfigureAwait(false);
    }

    private string GetPath(string key)
    {
        var safeKey = string.Concat(key.Select(c =>
            Path.GetInvalidFileNameChars().Contains(c) ? '_' : c));

        return Path.Combine(_baseDirectory, $"{safeKey}.json");
    }
}
