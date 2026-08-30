interface Runner {
  run(value: number): string;
  stop(force?: boolean): void;
}
type Config = { retries: number };
enum Mode { Active, Passive }
export class Service extends BaseService implements Runner {
  async run(value: number = 1): Promise<string> { return String(value); }
  stop(force = false): void { void force; }
}
function helper<T>(value: T): T { return value; }
function parse(value: string): Config { return JSON.parse(value); }
function validate(value: unknown): boolean { return value != null; }
const normalize = (value: string): string => value.trim();
