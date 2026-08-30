public interface IRunner {
    string Run(int value);
    void Stop(bool force);
}
public enum Mode { Active, Passive }
public record Payload(string Value);
public struct Worker { public void Work() { } }
public class Service : BaseService, IRunner {
    [Obsolete]
    public async Task<string> RunAsync(int value = 1) { return value.ToString(); }
    public void Stop(bool force) { }
}
public static class Utility { public static string Parse(string value) { return value; } }
