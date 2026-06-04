import sys
import os

# Add project root
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)
import pandas as pd

from src.data_processing import (
    CustomerFeatureBuilder
)


def test_customer_features_created():

    data = {
    "CustomerId": ["C1", "C1"],
    "Amount": [100, 200],
    "TransactionStartTime": [
        "2024-01-01",
        "2024-01-02"
    ],
    "ChannelId": ["A", "A"],
    "ProductCategory": ["airtime", "airtime"],
    "ProviderId": ["P1", "P1"],
    "ProductId": ["PR1", "PR1"],
    "PricingStrategy": [2, 2]
}
    df = pd.DataFrame(data)

    builder = CustomerFeatureBuilder()

    result = builder.fit_transform(df)

    assert (
        "total_transaction_amount"
        in result.columns
    )


def test_transaction_count():

    data = {
    "CustomerId": ["C1", "C1", "C1"],
    "Amount": [10, 20, 30],
    "TransactionStartTime": [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03"
    ],
    "ChannelId": ["A", "A", "A"],
    "ProductCategory": [
        "airtime",
        "airtime",
        "airtime"
    ],
    "ProviderId": ["P1", "P1", "P1"],
    "ProductId": ["PR1", "PR1", "PR1"],
    "PricingStrategy": [2, 2, 2]
}

    df = pd.DataFrame(data)

    builder = CustomerFeatureBuilder()

    result = builder.fit_transform(df)

    assert (
        result["transaction_count"].iloc[0]
        == 3
    )
    
    