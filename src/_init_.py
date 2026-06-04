import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def extract_time_features(df):
    ...
    return df


def create_customer_features(df):
    ...
    return customer_features


def build_feature_table(df):
    feature_df = df.copy()
    ...
    return feature_df

def create_preprocessing_pipeline(
    numerical_features,
    categorical_features
):
    ...
    return preprocessor
def process_data(df):
    ...
    return X_processed
if __name__ == "__main__":

    df = pd.read_excel("data/raw/data.xlsx")

    processed_data = build_feature_table(df)

    processed_data.to_csv(
        "data/processed/processed_data.csv",
        index=False
    )

    print("Processed data saved successfully.")

    