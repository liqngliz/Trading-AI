namespace Integrations.TwelveData;

public interface IRepository<T>
{
    Task<T?> GetAsync(string key, CancellationToken cancellationToken = default);
    Task SaveAsync(string key, T value, CancellationToken cancellationToken = default);
}
