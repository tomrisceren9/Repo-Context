public class Sample extends BaseService implements Runner {
    @Override
    public String run(int value) { return String.valueOf(value); }
    public void stop(boolean force) { }
}
interface Runner { String execute(int value); }
enum Mode { ACTIVE, PASSIVE }
record Payload(String value) { }
class Helper {
    static int add(int left, int right) { return left + right; }
    static int subtract(int left, int right) { return left - right; }
    String parse(String value) { return value; }
}
