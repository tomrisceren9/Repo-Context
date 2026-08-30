def process_data(value: str) -> str:
    """Generic processing helper intentionally unrelated to payment capture."""
    return value.strip()


def load_record(identifier: str) -> dict[str, str]:
    """Generic persistence helper intentionally unrelated to orders and users."""
    return {"id": identifier}
