using Integrations.TwelveData;

namespace FeatureEngine;


public interface IFeatureEngine 
{
    public Dictionary<DateTime, Dictionary<string, float?>> GetFeatureMatrix();
}

public class FeatureEngine : IFeatureEngine
{
    private readonly IRepository<IndicatorCacheDocument> _indicatorRepository;
    private readonly IRepository<TimeSeriesCacheDocument> _timeSeriesRepository;
    FeatureEngineConfig _config;
    DateTime _datasetStart;
    DateTime _datasetEnd;
    private readonly  Dictionary<DateTime, Dictionary<string, float?>> _featureMatrix = new Dictionary<DateTime, Dictionary<string, float?>>();

    public FeatureEngine(
        IRepository<IndicatorCacheDocument> indicatorRepository, 
        IRepository<TimeSeriesCacheDocument> timeSeriesRepository,
        FeatureEngineConfig config,
        DateTime datasetStart,
        DateTime datasetEnd
        )
    {
        _indicatorRepository = indicatorRepository ?? throw new ArgumentNullException(nameof(indicatorRepository));
        _timeSeriesRepository = timeSeriesRepository ?? throw new ArgumentNullException(nameof(timeSeriesRepository));    
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _datasetStart = datasetStart;
        _datasetEnd = datasetEnd;
    }

    public Dictionary<DateTime, Dictionary<string, float?>>GetFeatureMatrix()
    {
        throw new NotImplementedException();
    }
}