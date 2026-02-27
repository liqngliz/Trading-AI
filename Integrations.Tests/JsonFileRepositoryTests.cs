using System.Text.Json;
using Integrations.TwelveData;

namespace Integrations.Tests;

public class JsonFileRepositoryTests : IDisposable
{
    private readonly string _tempDir;

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
        Assert.Throws<ArgumentException>(() => new JsonFileRepository<object>(dir!));
    }

    [Fact]
    public void Constructor_ValidDirectory_CreatesDirectoryIfMissing()
    {
        var dir = Path.Combine(_tempDir, "sub");
        _ = new JsonFileRepository<object>(dir);
        Assert.True(Directory.Exists(dir));
    }

    // ── GetAsync ──────────────────────────────────────────────────────────────

    [Fact]
    public async Task GetAsync_FileDoesNotExist_ReturnsNull()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir);
        var result = await repo.GetAsync("nonexistent");
        Assert.Null(result);
    }

    [Fact]
    public async Task GetAsync_ValidJsonFile_DeserializesCorrectly()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir);
        var dto = new SimpleDto { Name = "test", Value = 42 };

        await repo.SaveAsync("mykey", dto);
        var result = await repo.GetAsync("mykey");

        Assert.NotNull(result);
        Assert.Equal("test", result!.Name);
        Assert.Equal(42, result.Value);
    }

    [Fact]
    public async Task GetAsync_InvalidJsonFile_ThrowsJsonException()
    {
        Directory.CreateDirectory(_tempDir);
        var path = Path.Combine(_tempDir, "badkey.json");
        await File.WriteAllTextAsync(path, "not valid json {{{");

        var repo = new JsonFileRepository<SimpleDto>(_tempDir);
        await Assert.ThrowsAsync<JsonException>(() => repo.GetAsync("badkey"));
    }

    // ── SaveAsync ─────────────────────────────────────────────────────────────

    [Fact]
    public async Task SaveAsync_WritesFileToExpectedPath()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir);
        await repo.SaveAsync("mykey", new SimpleDto { Name = "x", Value = 1 });

        var expectedPath = Path.Combine(_tempDir, "mykey.json");
        Assert.True(File.Exists(expectedPath));
    }

    [Fact]
    public async Task SaveAsync_OverwritesExistingFile()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir);
        await repo.SaveAsync("mykey", new SimpleDto { Name = "first", Value = 1 });
        await repo.SaveAsync("mykey", new SimpleDto { Name = "second", Value = 2 });

        var result = await repo.GetAsync("mykey");
        Assert.Equal("second", result!.Name);
        Assert.Equal(2, result.Value);
    }

    [Fact]
    public async Task SaveAsync_KeyWithInvalidFileNameChars_SanitizesKey()
    {
        // "XAU/USD" contains '/' which is invalid in a filename
        var repo = new JsonFileRepository<SimpleDto>(_tempDir);
        await repo.SaveAsync("XAU/USD", new SimpleDto { Name = "gold", Value = 1900 });

        // Should produce "XAU_USD.json"
        var expectedPath = Path.Combine(_tempDir, "XAU_USD.json");
        Assert.True(File.Exists(expectedPath));
    }

    [Fact]
    public async Task GetAsync_AfterSaveAsync_SameKeyWithSpecialChars_ReadsBack()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir);
        var dto = new SimpleDto { Name = "gold", Value = 1900 };

        await repo.SaveAsync("XAU/USD", dto);
        var result = await repo.GetAsync("XAU/USD");

        Assert.NotNull(result);
        Assert.Equal("gold", result!.Name);
    }

    [Fact]
    public async Task GetAsync_AfterSaveAsync_RoundTripPreservesObject()
    {
        var repo = new JsonFileRepository<SimpleDto>(_tempDir);
        var dto = new SimpleDto { Name = "roundtrip", Value = 999 };

        await repo.SaveAsync("rtkey", dto);
        var result = await repo.GetAsync("rtkey");

        Assert.NotNull(result);
        Assert.Equal(dto.Name, result!.Name);
        Assert.Equal(dto.Value, result.Value);
    }

    // ── Helper DTO ────────────────────────────────────────────────────────────

    private sealed class SimpleDto
    {
        public string Name { get; set; } = string.Empty;
        public int Value { get; set; }
    }
}
