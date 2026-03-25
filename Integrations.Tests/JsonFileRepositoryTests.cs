using System.Text.Json.Serialization.Metadata;
using Integrations.TwelveData;

namespace Integrations.Tests;

internal sealed class SimpleDto
{
    public string Name { get; set; } = string.Empty;
    public int Value { get; set; }
}

public class JsonFileRepositoryTests : IDisposable
{
    private readonly string _tempDir;
    private static JsonTypeInfo<SimpleDto> TypeInfo =>
        (JsonTypeInfo<SimpleDto>)System.Text.Json.JsonSerializerOptions.Default.GetTypeInfo(typeof(SimpleDto));

    public JsonFileRepositoryTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"JsonFileRepositoryTests_{Guid.NewGuid():N}");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    // ── Constructor ───────────────────────────────────────────────────────────

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Constructor_InvalidBaseDirectory_ThrowsArgumentException(string? dir)
    {
        Assert.Throws<ArgumentException>(() => new JsonFileRepository<SimpleDto>(dir!, TypeInfo));
    }

    [Fact]
    public void Constructor_ValidDirectory_CreatesDirectoryIfMissing()
    {
        var dir = Path.Combine(_tempDir, "sub");
        _ = new JsonFileRepository<SimpleDto>(dir, TypeInfo);
        Assert.True(Directory.Exists(dir));
    }

    // ── GetAsync ──────────────────────────────────────────────────────────────

    [Fact]
    public async Task GetAsync_FileDoesNotExist_ReturnsNull()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir, TypeInfo);
        var result = await repo.GetAsync("sym", "nonexistent");
        Assert.Null(result);
    }

    [Fact]
    public async Task GetAsync_ValidJsonFile_DeserializesCorrectly()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir, TypeInfo);
        var dto = new SimpleDto { Name = "test", Value = 42 };

        await repo.SaveAsync("sym", "mykey", dto);
        var result = await repo.GetAsync("sym", "mykey");

        Assert.NotNull(result);
        Assert.Equal("test", result!.Name);
        Assert.Equal(42, result.Value);
    }

    [Fact]
    public async Task GetAsync_InvalidJsonFile_DeletesFileAndReturnsNull()
    {
        Directory.CreateDirectory(_tempDir);
        // file name matches GetPath("sym", "badkey") = sym_badkey.json
        var path = Path.Combine(_tempDir, "sym_badkey.json");
        await File.WriteAllTextAsync(path, "not valid json {{{");

        var repo = new JsonFileRepository<SimpleDto>(_tempDir, TypeInfo);
        var result = await repo.GetAsync("sym", "badkey");

        Assert.Null(result);
        Assert.False(File.Exists(path));
    }

    // ── SaveAsync ─────────────────────────────────────────────────────────────

    [Fact]
    public async Task SaveAsync_WritesFileToExpectedPath()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir, TypeInfo);
        await repo.SaveAsync("sym", "mykey", new SimpleDto { Name = "x", Value = 1 });

        var expectedPath = Path.Combine(_tempDir, "sym_mykey.json");
        Assert.True(File.Exists(expectedPath));
    }

    [Fact]
    public async Task SaveAsync_OverwritesExistingFile()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir, TypeInfo);
        await repo.SaveAsync("sym", "mykey", new SimpleDto { Name = "first", Value = 1 });
        await repo.SaveAsync("sym", "mykey", new SimpleDto { Name = "second", Value = 2 });

        var result = await repo.GetAsync("sym", "mykey");
        Assert.Equal("second", result!.Name);
        Assert.Equal(2, result.Value);
    }

    [Fact]
    public async Task SaveAsync_KeyWithInvalidFileNameChars_SanitizesKey()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir, TypeInfo);
        await repo.SaveAsync("XAU/USD", "4h", new SimpleDto { Name = "gold", Value = 1900 });

        var expectedPath = Path.Combine(_tempDir, "XAU_USD_4h.json");
        Assert.True(File.Exists(expectedPath));
    }

    [Fact]
    public async Task GetAsync_AfterSaveAsync_SameKeyWithSpecialChars_ReadsBack()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir, TypeInfo);
        var dto = new SimpleDto { Name = "gold", Value = 1900 };

        await repo.SaveAsync("XAU/USD", "4h", dto);
        var result = await repo.GetAsync("XAU/USD", "4h");

        Assert.NotNull(result);
        Assert.Equal("gold", result!.Name);
    }

    [Fact]
    public async Task GetAsync_AfterSaveAsync_RoundTripPreservesObject()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir, TypeInfo);
        var dto = new SimpleDto { Name = "roundtrip", Value = 999 };

        await repo.SaveAsync("sym", "rtkey", dto);
        var result = await repo.GetAsync("sym", "rtkey");

        Assert.NotNull(result);
        Assert.Equal(dto.Name, result!.Name);
        Assert.Equal(dto.Value, result.Value);
    }
}
