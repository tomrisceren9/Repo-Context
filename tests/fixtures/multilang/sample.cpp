namespace application {
class Service : public BaseService {
public:
    void run(int value);
    void stop(bool force = false);
};
struct Worker { void execute(); };
enum class Mode { Active, Passive };
int add(int left, int right) { return left + right; }
int subtract(int left, int right) { return left - right; }
Service create_service() { return Service{}; }
void destroy_service(Service &service) { service.stop(true); }
}
