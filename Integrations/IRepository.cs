namespace Integrations.TwelveData;

public interface IRepository<T>
{
    Task<T?> GetAsync(string symbol, string interval, CancellationToken cancellationToken = default);
    Task SaveAsync(string symbol, string interval, T value, CancellationToken cancellationToken = default);
}
