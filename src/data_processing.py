
# src/data_processing.py

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class AggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        customer_features = (
            df.groupby("CustomerId")
            .agg(
                total_transaction_amount=("Amount", "sum"),
                avg_transaction_amount=("Amount", "mean"),
                transaction_count=("Amount", "count"),
                std_transaction_amount=("Amount", "std"),
                max_transaction_amount=("Amount", "max"),
                min_transaction_amount=("Amount", "min")
            )
            .reset_index()
        )

        return customer_features


class TimeFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        df["transaction_hour"] = df["TransactionStartTime"].dt.hour
        df["transaction_day"] = df["TransactionStartTime"].dt.day
        df["transaction_month"] = df["TransactionStartTime"].dt.month
        df["transaction_year"] = df["TransactionStartTime"].dt.year

        return df


class CustomerFeatureBuilder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        agg_df = (
            df.groupby("CustomerId")
            .agg(
                total_transaction_amount=("Amount", "sum"),
                avg_transaction_amount=("Amount", "mean"),
                transaction_count=("Amount", "count"),
                std_transaction_amount=("Amount", "std"),
                max_transaction_amount=("Amount", "max"),
                min_transaction_amount=("Amount", "min")
            )
            .reset_index()
        )

        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        df["transaction_hour"] = df["TransactionStartTime"].dt.hour
        df["transaction_day"] = df["TransactionStartTime"].dt.day
        df["transaction_month"] = df["TransactionStartTime"].dt.month
        df["transaction_year"] = df["TransactionStartTime"].dt.year

        time_df = (
            df.groupby("CustomerId")
            .agg(
                avg_transaction_hour=("transaction_hour", "mean"),
                avg_transaction_day=("transaction_day", "mean"),
                most_recent_month=("transaction_month", "max"),
                most_recent_year=("transaction_year", "max"),
                provider_id=("ProviderId","first"),
                product_id=("ProductId","first"),
                pricing_strategy=("PricingStrategy","first"),
                channel_id=("ChannelId", "first"),
                product_category=("ProductCategory", "first")
            )
            .reset_index()
        )

        customer_df = agg_df.merge(
            time_df,
            on="CustomerId",
            how="left"
        )

        return customer_df


class DataProcessor:

    def __init__(self):

        self.numeric_features = [
            "total_transaction_amount",
            "avg_transaction_amount",
            "transaction_count",
            "std_transaction_amount",
            "max_transaction_amount",
            "min_transaction_amount",
            "avg_transaction_hour",
            "avg_transaction_day",
            "avg_transaction_month",
            "avg_transaction_year"
        ]

        self.categorical_features = [
            "channel_id",
            "product_category",
             "provider_id",
             "product_id",
             "pricing_strategy"
        ]

        self.preprocessor = None

    def build_pipeline(self):

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    self.numeric_features
                ),
                (
                    "cat",
                    categorical_pipeline,
                    self.categorical_features
                )
            ]
        )

        return self.preprocessor

