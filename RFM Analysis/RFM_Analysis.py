#region What is RFM?

"""RFM: Recency, Frequency, Monetary

RFM (Recency, Frequency, Monetary) is a data segmentation method used to analyze customer behavior. It evaluates:
- Recency: How recently a customer made a purchase.
- Frequency: How often a customer makes purchases.
- Monetary: How much a customer spends.

This analysis helps businesses classify customers, understand their value, and design targeted marketing strategies to increase retention and revenue.

RFM scores classify customers based on:
1. Recency: How recently a customer purchased (lower days = higher score).
2. Frequency: How often a customer purchases (more frequent = higher score).
3. Monetary: How much a customer spends (higher spend = higher score).

Each metric is scored from 1 (low) to 5 (high), forming a three-digit score (e.g., 555 for best customers). This helps segment customers and create targeted strategies.


Creating segments using RFM scores involves grouping customers based on their RFM values to understand behaviors and target them effectively. Common segments include:
1. Best Customers (e.g., 555): Recent, frequent buyers with high spending.
2. Loyal Customers: Frequent buyers with high spending over time.
3. At-Risk Customers: Previously good customers who haven't purchased recently.
4. New Customers: Recent buyers with lower frequency but potential for growth.
5. Churned Customers (e.g., 111): Long inactive, low spenders.

"""



