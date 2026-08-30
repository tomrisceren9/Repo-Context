@sealed
class Service extends BaseService {
  async run(value = 1) { return value; }
  stop(force = false) { return force; }
}
class BaseService { start() { return true; } }
export function helper(value, fallback = 2) { return value ?? fallback; }
export function parse(value) { return JSON.parse(value); }
export function validate(value) { return value != null; }
export function* stream(items) { yield* items; }
export const normalize = (value) => value.trim();
