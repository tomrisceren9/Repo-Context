class OrderRepository:
    def find_order_by_id(self, order_id: str) -> dict[str, str] | None:
        """Load one purchase order using its persistent identifier."""
        return {"id": order_id}

    def list_recent_orders(self, limit: int = 20) -> list[dict[str, str]]:
        return []
