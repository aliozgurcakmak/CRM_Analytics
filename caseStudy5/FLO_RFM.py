###############################################################
# Customer Segmentation with RFM
###############################################################

import pandas as pd
import numpy as np
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

###############################################################
#region Business Problem
###############################################################
# FLO wants to divide its customers into segments and determine marketing strategies according to these segments.
# For this purpose, customer behaviors will be defined and groups will be created based on these behavioral clusters.
#endregion

###############################################################
#region Data Set Story
###############################################################

# The dataset consists of information obtained from past shopping behaviors of customers who made OmniChannel purchases 
# (both online and offline shopping) in 2020 - 2021.

# master_id: Unique customer number
# order_channel: Platform channel used for shopping (Android, ios, Desktop, Mobile, Offline)
# last_order_channel: Channel of the last purchase
# first_order_date: Date of customer's first purchase
# last_order_date: Date of customer's last purchase
# last_order_date_online: Date of customer's last online purchase
# last_order_date_offline: Date of customer's last offline purchase
# order_num_total_ever_online: Total number of online purchases by the customer
# order_num_total_ever_offline: Total number of offline purchases by the customer
# customer_value_total_ever_offline: Total amount paid by the customer in offline purchases
# customer_value_total_ever_online: Total amount paid by the customer in online purchases
# interested_in_categories_12: List of categories purchased by the customer in the last 12 months
#endregion

###############################################################
# TASKS
###############################################################

# TASK 1: Data Understanding and Preparation

#region 1. Read the flo_data_20K.csv file
df_ = pd.read_csv(r"C:\Users\alioz\OneDrive\Belgeler\datasets\FLOMusteriSegmentasyonu\flo_data_20k.csv")
df = df_.copy()
#endregion

#region 2. Examine the dataset for:
# a. First 10 observations
df.head(10)
# b. Variable names
df.info()
# c. Descriptive statistics
df.describe().T
# d. Missing values
df.isnull().sum()
# e. Variable types
df.dtypes
#endregion

#region # 3. Create new variables for total number of purchases and total spending for omnichannel customers
df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["TotalOrder"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
#endregion

#region # 4. Review variable types. Convert date variables to date type
df['first_order_date'] = pd.to_datetime(df['first_order_date'])
df['last_order_date'] = pd.to_datetime(df['last_order_date'])
df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])
#endregion

#region 5. Examine the distribution of customer numbers, average number of products, and average spending across shopping channels

df.groupby("order_channel").agg(
        customer_count=("master_id", "nunique"),
        total_order=("TotalOrder", "sum"),
        total_price=("TotalPrice", "sum"))

#endregion

#region 6. List the top 10 customers generating the most revenue
df.sort_values(by="TotalPrice", ascending=False).head(10)
#endregion

#region 7. List the top 10 customers with the most orders
df.sort_values(by="TotalOrder", ascending=False).head(10)
#endregion

#region 8. Functionalize the data preprocessing process
def preprocess_flo_data(df):
    """
    Preprocess FLO customer data by:
    1. Creating total price and total order variables
    2. Converting date columns to datetime
    3. Analyzing distribution across shopping channels
    4. Identifying top revenue and order customers

    Parameters:
    df (pandas.DataFrame): Input DataFrame with FLO customer data

    Returns:
    tuple: Processed DataFrame, channel distribution, top revenue customers, top order customers
    """

    df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df["TotalOrder"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]

    # Convert date columns to datetime
    date_columns = ['first_order_date', 'last_order_date',
                    'last_order_date_online', 'last_order_date_offline']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    [col for col in df.columns if col in date_columns ]

    # Analyze distribution across shopping channels
    channel_distribution = df.groupby("order_channel").agg(
        customer_count=("master_id", "nunique"),
        total_order=("TotalOrder", "sum"),
        total_price=("TotalPrice", "sum")
    )

    # Top 10 customers by revenue
    top_revenue_customers = df.sort_values(by="TotalPrice", ascending=False).head(10)

    # Top 10 customers by number of orders
    top_order_customers = df.sort_values(by="TotalOrder", ascending=False).head(10)

    return df, channel_distribution, top_revenue_customers, top_order_customers
#endregion


# TASK 2: Calculating RFM

#region Calculate RFM Metrics
df["last_order_date"].max()
today_date = dt.datetime(2021,6,1)

rfm = df.groupby("master_id").agg({
    "last_order_date": lambda date: (today_date - date.max()).days,
    "TotalOrder": lambda order: order.sum(),
    "TotalPrice": lambda price: price.sum()})

rfm.head()
rfm.columns = ["recency", "frequency", "monetary"]
rfm.describe().T
#endregion


# TASK 3: Calculating RF and RFM Scores

#region Calculate RF and RFM Scores
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["RF_Score"] = (rfm["recency_score"].astype(str) +  rfm["frequency_score"].astype(str))
rfm["RFM_Score"] = rfm["RF_Score"].astype(str) + rfm["monetary_score"].astype(str)
#endregion

# TASK 4: Defining Segments from RF Scores

#region Define the segments from RF Scores

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

rfm["segment"] = rfm["RF_Score"].replace(seg_map, regex=True)
rfm = rfm[["segment", "recency",  "frequency", "monetary"]]

#endregion

# TASK 5: Time for Action!

#region 1. Examine the recency, frequency, and monetary averages of segments
rfm.groupby("segment").agg({"recency": "mean",
                           "frequency": "mean",
                           "monetary": "mean"})
#endregion

# 2. Find customers with specific profiles using RFM analysis and save their customer IDs to csv

#region a. FLO is adding a new women's shoe brand.

rfm_final = rfm.merge(df[["master_id", "interested_in_categories_12"]], on="master_id", how="left")

new_brand_target_customer = rfm_final[(rfm_final["segment"].isin(["loyal_customers","champions"])) & (rfm_final["interested_in_categories_12"].str.contains("KADIN"))]

new_brand_target_customer_id = new_brand_target_customer["master_id"]

new_brand_target_customer_id.to_csv("new_brand_target_customer_id.csv")




#endregion

#region b. A discount of around 40% is planned for Men's and Children's products.

discount_target_customer = rfm_final[(rfm_final["segment"].isin(["cant_loose","hibernating", "new_customers"])) & (rfm_final["interested_in_categories_12"].str.contains("ERKEK|COCUK|AKTIFCOCUK", na=False))]

discount_target_customer_id = discount_target_customer["master_id"]

discount_target_customer_id.to_csv("discount_target_customer_id.csv")
#endregion


# TASK 6: Functionalize the entire process

def create_rfm_analysis(dataframe, csv_path=False):
    """
    Perform complete RFM analysis on customer data:
    1. Preprocess the data
    2. Calculate RFM metrics
    3. Create segments
    4. Generate target customer lists

    Parameters:
    -----------
    dataframe: pandas.DataFrame
        Input dataframe containing customer purchase data
    csv_path: bool, optional
        Whether to save target customer lists to CSV files

    Returns:
    --------
    tuple:
        processed_df: Processed dataframe with total metrics
        rfm: RFM dataframe with segments
        new_brand_targets: Target customers for new women's brand
        discount_targets: Target customers for men's/children's discount
    """

    # Copy the dataframe to avoid modifying the original
    df = dataframe.copy()

    # 1. Preprocess the data
    # Calculate totals
    df["TotalPrice"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df["TotalOrder"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]

    # Convert date columns
    date_columns = ['first_order_date', 'last_order_date',
                    'last_order_date_online', 'last_order_date_offline']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # 2. Calculate RFM metrics
    today_date = dt.datetime(2021, 6, 1)

    rfm = df.groupby("master_id").agg({
        "last_order_date": lambda date: (today_date - date.max()).days,
        "TotalOrder": lambda order: order.sum(),
        "TotalPrice": lambda price: price.sum()
    })

    rfm.columns = ["recency", "frequency", "monetary"]

    # 3. Calculate RFM scores
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RF_Score"] = (rfm["recency_score"].astype(str) +
                       rfm["frequency_score"].astype(str))

    # 4. Define segments
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["segment"] = rfm["RF_Score"].replace(seg_map, regex=True)

    # 5. Merge with original data for category information
    rfm_final = rfm.merge(df[["master_id", "interested_in_categories_12"]],
                          on="master_id", how="left")

    # 6. Generate target customer lists
    # New women's brand targets
    new_brand_targets = rfm_final[
        (rfm_final["segment"].isin(["loyal_customers", "champions"])) &
        (rfm_final["interested_in_categories_12"].str.contains("KADIN"))
        ]["master_id"]

    # Discount targets
    discount_targets = rfm_final[
        (rfm_final["segment"].isin(["cant_loose", "hibernating", "new_customers"])) &
        (rfm_final["interested_in_categories_12"].str.contains("ERKEK|COCUK|AKTIFCOCUK", na=False))
        ]["master_id"]

    # Save to CSV if requested
    if csv_path:
        new_brand_targets.to_csv("new_brand_target_customer_id.csv", index=False)
        discount_targets.to_csv("discount_target_customer_id.csv", index=False)

    return df, rfm_final, new_brand_targets, discount_targets

# Example usage:
# df_ = pd.read_csv("flo_data_20k.csv")
# processed_df, rfm_results, new_brand_targets, discount_targets = create_rfm_analysis(df_, csv_path=True)

create_rfm_analysis(df, csv_path=True)

df = df_.copy()