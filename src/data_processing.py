def create_rfm_features(df):
    """
    Create Recency, Frequency and Monetary metrics
    for each customer.
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
from sklearn.preprocessing import StandardScaler
def scale_rfm(rfm):

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[[
            "Recency",
            "Frequency",
            "Monetary"
        ]]
    )

    return rfm_scaled
from sklearn.cluster import KMeans
def cluster_customers(rfm):

    rfm_scaled = scale_rfm(rfm)
    rfm = build_rfm_table(df)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    return rfm
rfm_clustered = cluster_customers(rfm)

print(
    rfm_clustered.groupby("cluster")[
        ["Recency", "Frequency", "Monetary"]
    ].mean()
)
