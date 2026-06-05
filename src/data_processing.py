import os
import pandas as pd

# Extracted schema headers from your Xente dataset
COLUMN_NAMES = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
    "FraudResult"
]

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Xente Fraud Detection dataset.
    Ensures correct header mapping and automatically parses date string columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file could not be found at: '{file_path}'")
        
    try:
        # Load standard comma-separated file, parse dates, and align column names
        df = pd.read_csv(
            file_path, 
            sep=',', 
            parse_dates=['TransactionStartTime'], # Automatically optimizes the temporal string
            infer_datetime_format=True
        )
        return df
        
    except Exception as e:
        raise IOError(f"Failed to cleanly load the Xente CSV. Details: {e}")
