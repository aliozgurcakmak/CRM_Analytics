###############################
""" CUSTOMER LIFETIME VALUE """
###############################

# Data Preparation
# Average Order Value (average_order_value = total_price / total_transaction)
# Purchase Frequency (total_transaction / total_number_of_customers)
# Repeat Rate & Churn Rate (number of customers who made multiple purchases / all customers)
# Profit Margin (profit_margin = total_price * 0.10)
# Customer Value (customer_value = average_order_value * purchase_frequency)
# Customer Lifetime Value (CLTV = (customer_value / churn_rate) * profit_margin)
# Creating Segments
# BONUS: Functioning all operations

#region Data Preparation
import pandas as pd
import numpy as np
from joblib import PrintTime
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel(r"C:\Users\alioz\Documents\crmAnalytics-221211-020816\crmAnalytics\datasets\online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()

df.isnull().sum()
df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]

df.describe().T
print("                      ")
df = df[df["Quantity"] > 0]

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                        "Quantity": lambda x: x.sum(),
                                        "TotalPrice": lambda x: x.sum()})
cltv_c.columns = ["total_transaction", "total_unit", "total_price"]
#endregion

#region Average Order Value (average_order_value = total_price / total_transaction)
cltv_c.head()
cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]
#endregion

#region Purchase Frequency (total_transaction / total_number_of_customers)
cltv_c["total_transaction"]
cltv_c.shape[0]
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
#endregion

#region Repeat Rate & Churn Rate (number of customers who made multiple purchases / all customers)

repeat_rate = cltv_c[cltv_c["total_transaction"]>1].shape[0] / cltv_c.shape[0]
churnRate = 1 - repeat_rate

#endregion

#region Profit Margin (profit_margin = total_price * 0.10)

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

#endregion

#region Customer Value (customer_value = average_order_value * purchase_frequency)

cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

#endregion

#region Customer Lifetime Value (CLTV = (customer_value / churn_rate) * profit_margin)
cltv_c["cltv"] = (cltv_c["customer_value"] / churnRate) * cltv_c["profit_margin"]
cltv_c.sort_values("cltv", ascending=False).head()

#endregion

#region Creating Segments
cltv_c.sort_values("cltv", ascending=False).tail()
cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_c.sort_values("cltv", ascending=False).head()

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_csv("cltv_c.csv")
#endregion

#region  BONUS: Functioning all operations

def create_cltv_c(dataframe, profit=10):
    # Data Preparation
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                                   "Quantity": lambda x: x.sum(),
                                                   "TotalPrice": lambda x: x.sum()})
    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

    # avg_order_value
    cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churnRate = 1 - repeat_rate

    # profit_margin
    cltv_c["profit_margin"] = cltv_c["total_price"] * profit

    # Customer Value
    cltv_c["customer_value"] = (cltv_c["average_order_value"] * cltv_c["purchase_frequency"])

    # Customer Lifetime Value
    cltv_c["cltv"] = (cltv_c["customer_value"] / churnRate) * cltv_c["profit_margin"]

    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c
df = df_.copy()

clv = create_cltv_c(df)

#endregion