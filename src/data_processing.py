import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================================
# TASK 3: FEATURE ENGINEERING
# ==========================================================

def extract_time_features(df):
    """
    Extract time-based features from TransactionStartTime.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["transaction_day"] = (
        df["TransactionStartTime"].dt.day
    )

    df["transaction_month"] = (
        df["TransactionStartTime"].dt.month
    )

    df["transaction_year"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


def create_customer_features(df):
    """
    Create customer-level aggregate features.
    """

    customer_features = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
            total_value=("Value", "sum")
        )
        .reset_index()
    )

    customer_features["std_transaction_amount"] = (
        customer_features["std_transaction_amount"]
        .fillna(0)
    )

    return customer_features


def build_feature_table(df):
    """
    Merge aggregate features with transaction data.
    """

    df = extract_time_features(df)

    customer_features = create_customer_features(df)

    feature_df = df.merge(
        customer_features,
        on="CustomerId",
        how="left"
    )

    return feature_df


# ==========================================================
# TASK 4: RFM FEATURE CREATION
# ==========================================================

def create_rfm_features(df):
    """
    Create Recency, Frequency and Monetary metrics.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm


def cluster_customers(rfm):
    """
    Cluster customers using KMeans.
    """

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    return rfm


def identify_high_risk_cluster(rfm):
    """
    Identify the cluster that represents
    the least engaged customers.
    """

    cluster_summary = (
        rfm.groupby("cluster")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean")
        )
    )

    print("\nCluster Summary")
    print(cluster_summary)

    high_risk_cluster = (
        cluster_summary["Frequency"]
        .idxmin()
    )

    print(
        f"\nSelected High Risk Cluster: "
        f"{high_risk_cluster}"
    )

    return high_risk_cluster


def create_risk_labels(rfm):
    """
    Create binary risk labels.
    """

    high_risk_cluster = (
        identify_high_risk_cluster(rfm)
    )

    rfm["is_high_risk"] = np.where(
        rfm["cluster"] == high_risk_cluster,
        1,
        0
    )

    return rfm


def add_target_variable(df):
    """
    Merge high-risk labels back to dataset.
    """

    rfm = create_rfm_features(df)

    rfm = cluster_customers(rfm)

    rfm = create_risk_labels(rfm)

    df = df.merge(
        rfm[
            [
                "CustomerId",
                "is_high_risk"
            ]
        ],
        on="CustomerId",
        how="left"
    )

    return df


# ==========================================================
# MAIN PROCESSING FUNCTION
# ==========================================================

def process_data(df):
    """
    Full data processing pipeline.
    """

    feature_df = build_feature_table(df)

    feature_df = add_target_variable(
        feature_df
    )

    return feature_df


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    INPUT_PATH = "data/raw/data.xlsx"
    OUTPUT_PATH = (
        "data/processed/processed_data.csv"
    )

    print("Loading raw dataset...")

    df = pd.read_excel(INPUT_PATH)

    print(
        f"Dataset Shape: {df.shape}"
    )

    processed_df = process_data(df)

    print("\nTarget Distribution")

    print(
        processed_df["is_high_risk"]
        .value_counts()
    )

    processed_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nProcessed dataset saved to:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        f"\nFinal Shape: "
        f"{processed_df.shape}"
    )

#     #Loading raw dataset...
# Dataset Shape: (95662, 16)

# Cluster Summary
#            Recency    Frequency      Monetary
# cluster                                      
# 0        61.877279     7.720196  8.172068e+04
# 1        12.726566    34.800000  2.725741e+05
# 2        29.000000  4091.000000 -1.049000e+08

# Selected High Risk Cluster: 0

# Target Distribution
# is_high_risk
# 0    84653
# 1    11009
# Name: count, dtype: int64

# Processed dataset saved to:
# data/processed/processed_data.csv

# Final Shape: (95662, 26)