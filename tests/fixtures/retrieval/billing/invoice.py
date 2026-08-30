def calculate_invoice_total(line_items: list[float], tax_rate: float) -> float:
    """Compute the final payable invoice amount including tax."""
    subtotal = sum(line_items)
    return subtotal * (1 + tax_rate)


def format_invoice_number(sequence: int) -> str:
    return f"INV-{sequence:08d}"
