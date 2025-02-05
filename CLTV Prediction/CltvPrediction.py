#################################################
"""CLTV Prediction with BG-NBD and Gamma-Gamma"""
#################################################

# Data Preparation
# Expected Number of Transaction with BG-NBD
# Expected Average Profit with Gamma-Gamma Model
# Calculation of CLTV with BG-NBD and Gamma-Gamma
# Creation of Segments according to CLTV
# Functioning whole Process


#region Data Preparation
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



df_ = pd.read_excel(r"C:\Users\alioz\Documents\crmAnalytics-221211-020816\crmAnalytics\datasets\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df_.describe().T

df.isnull().sum()
df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]

df.describe().T
print("                      ")
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T
df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011,12,11)
#endregion

#region Preparation Lifetime Data Structure

#recency = Time since the last purchase for the user.
#T: Age of Customer, Weekly (How long before the analysis date was the first purchase made?)
#frequency: Total number of recurring purchases (frequency must be bigger than 1)
#monetary: Average earning per purchase

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [
lambda date: (date.max() - date.min()).days,
lambda date: (today_date - date.min()).days],
"Invoice": lambda date: date.nunique(),
"TotalPrice": lambda price: price.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df.describe().T

cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
#endregion

#region Establishment of BG-NBD Model

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

### Who are the 10 customers we expect to purchase the most within 1 week?

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

bgf.predict(1, cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"] = bgf.predict(1,
 cltv_df["frequency"],
 cltv_df["recency"],
cltv_df["T"])

cltv_df["expected_purc_1_month"] = bgf.predict(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

bgf.predict(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sum()

##
bgf.predict(4*3, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"]).sum()
cltv_df["expected_purc_3_month"] = bgf.predict(4*3, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

## Eveluate of Predict Results

plot_period_transactions(bgf)
plt.show()
#endregion

#region Establishment of Gamma-Gamma Model

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

ggf.conditional_expected_average_profit( cltv_df["frequency"], cltv_df["monetary"]).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

#endregion

#region Calculation of CLTV with BG-NBD and Gamma-Gamma
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"], cltv_df["recency"], cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=3,
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()
cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values("clv", ascending=False).head(10)
#endregion

#region Creation of Segments according to CLTV

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(10)
cltv_final.groupby("segment").agg({"count", "mean", "sum"})
#endregion

#region Functioning whole Process

def create_cltv_p(dataframe, month = 3):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011,12,11)

    cltv_df = dataframe.groupby("Customer ID").agg({"InvoiceDate": [
    lambda date: (date.max() - date.min()).days,
    lambda date: (today_date - date.min()).days],
    "Invoice": lambda date: date.nunique(),
    "TotalPrice": lambda price: price.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ["recency", "T", "frequency", "monetary"]
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[cltv_df["frequency"] > 1]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df["frequency"],
                                                     cltv_df["recency"],
                                                     cltv_df["T"])

    cltv_df["expected_purc_1_month"] = bgf.predict(4, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

    cltv_df["expected_purc_3_month"] = bgf.predict(4*3, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df["frequency"], cltv_df["recency"], cltv_df["T"],
                                       cltv_df["monetary"],
                                       time=month,
                                       freq="W",
                                       discount_rate=0.01)

    cltv = cltv.reset_index()

    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final
df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")