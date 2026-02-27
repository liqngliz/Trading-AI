using System.Net;
using System.Text;

namespace Integrations.Tests.Helpers;

public sealed class FakeHttpMessageHandler : HttpMessageHandler
{
    private readonly Queue<HttpResponseMessage> _responses = new();

    public List<HttpRequestMessage> SentRequests { get; } = new();

    public void EnqueueResponse(HttpStatusCode statusCode, string content, string mediaType = "application/json")
    {
        _responses.Enqueue(new HttpResponseMessage(statusCode)
        {
            Content = new StringContent(content, Encoding.UTF8, mediaType)
        });
    }

    protected override Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request,
        CancellationToken cancellationToken)
    {
        SentRequests.Add(request);

        if (_responses.Count == 0)
            throw new InvalidOperationException("No more queued responses. Enqueue a response before making a request.");

        return Task.FromResult(_responses.Dequeue());
    }
}
