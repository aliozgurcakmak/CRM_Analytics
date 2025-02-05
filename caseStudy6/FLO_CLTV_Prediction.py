##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################################
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
###############################################################

# Business Problem
###############################################################
# FLO wants to determine a roadmap for sales and marketing activities.
# The company needs to predict the potential future value of existing customers to make medium and long-term plans.


###############################################################
# Dataset Story
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases
# as OmniChannel (both online and offline) in 2020-2021.

# master_id: Unique customer number
# order_channel: Which channel of the platform is used for shopping (Android, ios, Desktop, Mobile, Offline)
# last_order_channel: The channel where the most recent purchase was made
# first_order_date: Date of the customer's first purchase
# last_order_date: Date of the customer's last purchase
# last_order_date_online: Customer's last purchase date on the online platform
# last_order_date_offline: Customer's last purchase date on the offline platform
# order_num_total_ever_online: Total number of purchases made by the customer on the online platform
# order_num_total_ever_offline: Total number of purchases made by the customer offline
# customer_value_total_ever_offline: Total amount paid by the customer for offline purchases
# customer_value_total_ever_online: Total amount paid by the customer for online purchases
# interested_in_categories_12: List of categories the customer has shopped in last 12 months


###############################################################
# TASKS
###############################################################
# TASK 1: Data Preparation

#region 1. Read the flo_data_20K.csv file. Create a copy of the dataframe.
df_ = pd.read_csv(r"C:\Users\alioz\OneDrive\Belgeler\datasets\flo_data_20k.csv")
df = df_.copy()
df.head()
df_.describe().T
df.describe().T
#endregion

#region 2. Define the outlier_thresholds and replace_with_thresholds functions needed to suppress outlier values.
# Note: Frequency values must be integer when calculating CLTV. Round the upper and lower limits using round().

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range

    # Yuvarlama ekleniyor (integer olması gereken değişkenler için)
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#endregion

#region 3. Suppress outlier values, if any, for the variables "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

#endregion

#region 4. Create new variables for each customer's total number of purchases and spending across platforms.
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["total_customer_order"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]

#endregion

#region 5. Examine variable types. Convert date-expressing variables to date type.

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])


#endregion

# TASK 2: Creating CLTV Data Structure

#region 1. Take 2 days after the last shopping date in the dataset as the analysis date.

df["last_order_date"].max()
today_date = dt.datetime(2021,6,1)

#endregion

#region 2. Create a new CLTV dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.

 # Monetary value will be expressed as average value per purchase, recency and tenure values will be expressed in weekly terms.

cltv_df = (pd.DataFrame(
    {"customer_id": df["master_id"],
 "recency_cltv_weekly": ((df["last_order_date"] -
 df["first_order_date"]).dt.days) / 7,
 "T_weekly": ((today_date - df["first_order_date"]).dt.days) / 7,
 "frequency": df["total_customer_order"],
"monetary_cltv_avg": df["total_customer_value"] / df["total_customer_order"]}))
#endregion


# TASK 3: Establishing BG/NBD, Gamma-Gamma Models, Calculating CLTV
#region 1. Fit the BG/NBD model.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])
# a. Predict expected purchases from customers within 3 months and add to CLTV dataframe as exp_sales_3_month.

cltv_df["exp_sales_3_month"] = bgf.predict(12, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

# b. Predict expected purchases from customers within 6 months and add to CLTV dataframe as exp_sales_6_month.

cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

#endregion

#region 2. Fit the Gamma-Gamma model. Predict customers' expected average value and add to CLTV dataframe as exp_average_value.

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"]).sort_values(ascending=False).head(10)

#endregion

#region 3. Calculate 6-month CLTV and add to dataframe as cltv.
calc_cltv = ggf.customer_lifetime_value(bgf,
                                         cltv_df["frequency"],
                                         cltv_df["recency_cltv_weekly"],
                                         cltv_df["T_weekly"],
                                         cltv_df["monetary_cltv_avg"],
                                         time=6,
                                        freq="W",
                                        discount_rate = 0.01)

cltv_df["cltv"] = calc_cltv


# b. Observe the top 20 customers with highest CLTV values.
cltv_df.sort_values(by="cltv", ascending=False).head(20)
cltv_df.reset_index()


#endregion

# TASK 4: Creating Segments Based on CLTV

#region 1. Divide all your 6-month customers into 4 groups (segments) and add group names to the dataset. Add to dataframe as cltv_segment.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, ["D", "C", "B", "A"])

# 2. Provide brief 6-month action recommendations to management for 2 groups of your choice from the 4 groups

#region 1. VIP Programme for High Value and Frequent Shoppers
"""1. VIP Programme for High Value and Frequent Shoppers

-- Only those in segment A are eligible.

-- Customers must be frequent shoppers.


-- Monetary value must be above average (more than 75% of the monetary value)"""

cltv_df["monetary_cltv_avg"].describe()
# 75%  182.4500

# In the ‘A’ segment, we select customers who shop frequently and spend more than 75 percent of the average
vip_customers = cltv_df[(cltv_df['cltv_segment'] == 'A') &
                   (cltv_df['frequency'] > cltv_df['frequency'].median()) &
                   (cltv_df['monetary_cltv_avg'] > 182.4500)]

# We save VIP customer ids to CSV file
vip_customers['customer_id'].to_csv('vip_program_customers.csv', index=False)

#endregion


#region 4. Welcome Campaign for New Customers
"""Welcome Campaign for New Customers
-- Customer must have started shopping recently.
-- Customer must have made a few purchases but not be a loyal customer.
"""


# Filtering accoring to criterias
recent_threshold = 30  # Must have made a purchase in the last 30 days
frequency_threshold = 2  # Must have made at least 2 purchases.
loyal_segment = ['A', 'B']  # Loyal Customers

# We are choosing new customers.
new_customers = cltv_df[(cltv_df['recency_cltv_weekly'] <= recent_threshold) &
                   (cltv_df['frequency'] >= frequency_threshold) &
                   (~cltv_df['cltv_segment'].isin(loyal_segment))]

new_customers['customer_id'].to_csv('new_customer_welcome_campaign.csv', index=False)
#endregion
