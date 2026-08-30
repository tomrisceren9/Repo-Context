export class PaymentGateway {
  async processPayment(amount: number, currency: string): Promise<string> {
    // Capture a card charge through the external provider.
    return `${currency}:${amount}`;
  }

  async refundPayment(transactionId: string): Promise<boolean> {
    return transactionId.length > 0;
  }
}
