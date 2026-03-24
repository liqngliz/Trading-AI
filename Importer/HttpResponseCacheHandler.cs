using System.Net;
using System.Security.Cryptography;
using System.Text;

namespace Importer;

/// <summary>
/// A delegating handler that optionally serves HTTP responses from a local file cache.
/// When <see cref="UseCache"/> is true, a cached response body is returned if available
/// and any new responses are saved. When false, requests pass straight through.
/// </summary>
public sealed class HttpResponseCacheHandler : DelegatingHandler
{
    private readonly string _cacheDirectory;

    public bool UseCache { get; set; }

    public HttpResponseCacheHandler(string cacheDirectory, bool useCache = false)
    {
        if (string.IsNullOrWhiteSpace(cacheDirectory))
            throw new ArgumentException("Cache directory is required.", nameof(cacheDirectory));

        _cacheDirectory = cacheDirectory;
        UseCache = useCache;
        Directory.CreateDirectory(_cacheDirectory);
    }

    protected override async Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request,
        CancellationToken cancellationToken)
    {
        if (!UseCache)
            return await base.SendAsync(request, cancellationToken).ConfigureAwait(false);

        var key  = GetCacheKey(request.RequestUri);
        var path = Path.Combine(_cacheDirectory, $"{key}.json");

        if (File.Exists(path))
        {
            var body = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/json")
            };
        }

        var response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);

        if (response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            await File.WriteAllTextAsync(path, body, cancellationToken).ConfigureAwait(false);
            response.Content = new StringContent(body, Encoding.UTF8, "application/json");
        }

        return response;
    }

    private static string GetCacheKey(Uri? uri)
    {
        var raw = uri?.ToString() ?? string.Empty;
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(raw));
        return Convert.ToHexString(hash)[..16].ToLowerInvariant();
    }
}
