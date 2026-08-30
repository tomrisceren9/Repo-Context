def component(name: str):
    return lambda value: value


def route(path: str):
    return lambda value: value


class ServiceMeta(type):
    pass


class BaseService:
    pass


@component("core")
class Service(BaseService, metaclass=ServiceMeta):
    @route("/run")
    async def run(self, value: int = 1, /, *, verbose: bool = False) -> str:
        return str(value)

    def stop(self, *, force: bool = False) -> None:
        return None


def helper(value: str = "default") -> str:
    return value


def parse(value: str) -> dict[str, str]:
    return {"value": value}


def validate(value: object) -> bool:
    return value is not None


def build(*items: str) -> tuple[str, ...]:
    return items
