public class StockManager {
    public bool ReserveInventory(string sku, int quantity) {
        // Hold warehouse stock before an order is confirmed.
        return quantity > 0;
    }

    public void ReleaseInventory(string sku, int quantity) { }
}
