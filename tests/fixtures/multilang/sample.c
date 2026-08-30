struct Service { int value; };
union Payload { int integer; char *text; };
enum State { STATE_ON, STATE_OFF };
int add(int left, int right) { return left + right; }
int subtract(int left, int right) { return left - right; }
int multiply(int left, int right) { return left * right; }
int divide(int left, int right) { return right ? left / right : 0; }
void reset(struct Service *service) { service->value = 0; }
struct Service create_service(void) { return (struct Service){0}; }
void destroy_service(struct Service *service) { (void)service; }
