####################################
"""Costumer Segmentation with RFM"""
####################################
from itertools import groupby

# Business Problem
# Data Understanding
# Data Preparation
# Calculating RFM Metrics
# Calculating RFM Scores
# Creating and Analysing RFM Segments
# Functioning Whole Process


#region Business Problem

### An E-Commercial Company want to seperate their customers to segments and determine their commercing strategies accorindg to these segments.

### The dataset contains the sales of a UK-based online store between 01/12/2009 and 09/12/2011.

import pandas as pd
df = pd.read_excel(r"C:\Users\alioz\Documents\crmAnalytics-221211-020816\crmAnalytics\datasets\online_retail_II.xlsx")
#endregion

#region Data Understanding

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel(r"C:\Users\alioz\Documents\crmAnalytics-221211-020816\crmAnalytics\datasets\online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape
df.isnull().sum()
df["Description"].nunique()
df["Description"].value_counts()
df["Description"].value_counts().head()
df.groupby("Description").agg({"Quantity":"sum"}).head()
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()
df["Invoice"].nunique()
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.groupby("Invoice").agg({"TotalPrice":"sum"}).head()
df.groupby("Invoice").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending=False).head()

#endregion

#region Data Preparation

df.shape

df.isnull().sum()

df.dropna(inplace=True)
df.shape

df.describe().T
df = df[~df["Invoice"].str.contains("C", na=False)]

#endregion

#region Calculating RFM Metrics

# Recency, Frequency, Monetary

df.head()
today_date = dt.datetime(2010,12,11)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
 "Invoice": lambda Invoice: Invoice.nunique(),
  "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.head()
rfm.columns = ["Recency", "Frequency", "Monetary"]
rfm.describe().T

rfm = rfm[rfm["Monetary"] > 0]

#endregion

#region Calculating RFM Scores

rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_Score"] = (rfm["recency_score"].astype(str) +  rfm["frequency_score"].astype(str))

rfm.describe().T

champions = rfm[rfm["RFM_Score"] == "55"]
hibernaters = rfm[rfm["RFM_Score"] == "11"]

#endregion

#region Creating and Analysing RFM Segments

### regex

seg_map = {r'[1-2][1-2]': 'hibernating',
r'[1-2][3-4]': 'at_Risk',
r'[1-2]5': 'cant_loose',
r'3[1-2]': 'about_to_sleep',
r'33': 'need_attention',
r'[3-4][4-5]': 'loyal_customers',
r'41': 'promising',
r'51': 'new_customers',
r'[4-5][2-3]': 'potential_loyalists',
r'5[4-5]': 'champions'}

rfm["segment"] = rfm["RFM_Score"].replace(seg_map, regex=True)
rfm[["segment", "Recency",  "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "need_attention"].head()
rfm[rfm["segment"] == "cant_loose"].head()
rfm[rfm["segment"] == "cant_loose"].index

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

new_df.to_csv("new_customers.csv")
#endregion

#region Functioning Whole Process

def create_rfm(dataframe, csv=False):

 # DATA PREPARATION
 dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
 dataframe.dropna(inplace=True)
 dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

 # CALCULATING RFM METRICS
 today_date = dt.datetime(2011,12,11)
 rfm = dataframe.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
 "Invoice": lambda Invoice: Invoice.nunique(),
  "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
 rfm.columns = ["Recency", "Frequency", "Monetary"]
 rfm = rfm[rfm["Monetary"] > 0]

 # CALCULATING RFM SCORES
 rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
 rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
 rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

 # SCORES BE TRANSFORMED TO CATEGORIC VAL AND ADDED TO DF
 rfm["RFM_Score"] = (rfm["recency_score"].astype(str) +  rfm["frequency_score"].astype(str))

 # NAMING SEGMENTS
 seg_map = {r'[1-2][1-2]': 'hibernating',
           r'[1-2][3-4]': 'at_Risk',
                        r'[1-2]5': 'cant_loose',
                        r'3[1-2]': 'about_to_sleep',
                        r'33': 'need_attention',
                        r'[3-4][4-5]': 'loyal_customers',
                        r'41': 'promising',
                        r'51': 'new_customers',
                        r'[4-5][2-3]': 'potential_loyalists',
                        r'5[4-5]': 'champions'}
 rfm["segment"] = rfm["RFM_Score"].replace(seg_map, regex=True)
 rfm = rfm[["segment", "Recency",  "Frequency", "Monetary"]]
 rfm.index = rfm.index.astype(int)
 if csv:
  rfm.to_csv("rfm_scores.csv")

 return rfm

df = df_.copy()
rfm_new = create_rfm(df, csv=True)
#endregion


