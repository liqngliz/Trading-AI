using Integrations.TwelveData;

namespace FeatureEngine;

/// <summary>
/// Which indicators to fetch and at which intervals for a given symbol.
/// </summary>
public sealed record SymbolFetchConfig(
    string Symbol,
    string[] Intervals,
    TwelveDataEndpoint[] Indicators);

/// <summary>
/// Full configuration for a FeatureEngine dataset build.
/// </summary>
public sealed record FeatureEngineConfig(
    /// <summary>The symbol whose future return is the prediction target.</summary>
    string TargetSymbol,
    /// <summary>The base interval used for target bar alignment (e.g. "1h").</summary>
    string BaseInterval,
    /// <summary>How many base-interval bars ahead to predict.</summary>
    int[] TargetHorizons,
    /// <summary>All symbols and their indicator/interval fetch requirements.</summary>
    SymbolFetchConfig[] Features);
