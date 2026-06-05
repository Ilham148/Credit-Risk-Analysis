import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Extracted schema headers from your Xente dataset
COLUMN_NAMES = [
    "TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId",
    "CurrencyCode", "CountryCode", "ProviderId", "ProductId", "ProductCategory",
    "ChannelId", "Amount", "Value", "TransactionStartTime", "PricingStrategy", "FraudResult"
]


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Loads the Xente dataset and parses timestamps cleanly."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file could not be found at: '{file_path}'")
    try:
        return pd.read_csv(file_path, sep=',', parse_dates=['TransactionStartTime'])
    except Exception as e:
        raise IOError(f"Failed to cleanly load the Xente CSV. Details: {e}")


def engineering_rfm_proxy_targets(df_raw: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Computes customer RFM metrics, applies K-Means clustering, automatically
    identifies the lowest-engagement cluster, and builds the 'is_high_risk' target.
    """
    df = df_raw.copy()
    
    # Ensure timestamps are ready and set a unified snapshot baseline
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    # Extract Recency, Frequency, and Monetary scores per unique customer
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days, # Recency
        'TransactionId': 'count',                                         # Frequency
        'Amount': 'sum'                                                   # Monetary Value
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    # Scale values to prevent features with wide ranges from overpowering K-Means
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # Fit K-Means cluster to segment customers into 3 distinct behavioral pools
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Programmatically locate the high-risk cluster profile 
    # High Risk = Lowest engagement (characterized by Low Frequency and Low Monetary volume)
    cluster_profiles = rfm.groupby('Cluster').agg({
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # Identify the cluster index that has the worst combined interaction footprint
    high_risk_cluster = (cluster_profiles['Frequency'] * cluster_profiles['Monetary']).idxmin()
    
    # Construct the credit risk binary indicator column
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    # Map the proxy values cleanly back to the core transaction records
    target_mapping = rfm['is_high_risk'].to_dict()
    df_raw_with_proxy = df_raw.copy()
    df_raw_with_proxy['is_high_risk'] = df_raw_with_proxy['CustomerId'].map(target_mapping)
    
    print(f"📊 Proxy Engineering Completed: Flagged {rfm['is_high_risk'].sum()} / {len(rfm)} unique customers as High-Risk.")
    return df_raw_with_proxy


class XenteFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.customer_aggs_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        if 'CustomerId' in df.columns and 'Amount' in df.columns:
            group = df.groupby('CustomerId')['Amount']
            self.customer_aggs_ = {
                'total': group.sum().to_dict(),
                'avg': group.mean().to_dict(),
                'count': group.count().to_dict(),
                'std': group.std().fillna(0).to_dict()
            }
        return self

    def transform(self, X):
        df = X.copy()
        if 'CustomerId' in df.columns:
            df['Total_Amount_Cust'] = df['CustomerId'].map(self.customer_aggs_.get('total', {})).fillna(df['Amount'])
            df['Avg_Amount_Cust'] = df['CustomerId'].map(self.customer_aggs_.get('avg', {})).fillna(df['Amount'])
            df['Transaction_Count_Cust'] = df['CustomerId'].map(self.customer_aggs_.get('count', {})).fillna(1)
            df['Std_Amount_Cust'] = df['CustomerId'].map(self.customer_aggs_.get('std', {})).fillna(0)
        else:
            df['Total_Amount_Cust'] = df['Amount']
            df['Avg_Amount_Cust'] = df['Amount']
            df['Transaction_Count_Cust'] = 1
            df['Std_Amount_Cust'] = 0

        if 'TransactionStartTime' in df.columns:
            times = pd.to_datetime(df['TransactionStartTime'])
            df['TransactionHour'] = times.dt.hour
            df['TransactionDay'] = times.dt.day
            df['TransactionMonth'] = times.dt.month
            df['TransactionYear'] = times.dt.year
        else:
            df['TransactionHour'] = 0
            df['TransactionDay'] = 1
            df['TransactionMonth'] = 1
            df['TransactionYear'] = 2026

        drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']
        return df.drop(columns=[c for c in drop_cols if c in df.columns])


class NativeWoETransformer(BaseEstimator, TransformerMixin):
    """
    A robust, native Pandas Weight of Evidence (WoE) transformer.
    Replaces the broken xverse package dependency completely.
    """
    def __init__(self, bins=5):
        self.bins = bins
        self.woe_maps_ = {}
        self.iv_scores_ = {}

    def fit(self, X, y):
        if y is None:
            raise ValueError("WoE calculation requires an explicit binary target vector 'y'.")
        
        df = pd.DataFrame(X).copy()
        y_arr = np.array(y)
        total_pos = np.sum(y_arr == 1)
        total_neg = np.sum(y_arr == 0)

        total_pos = total_pos if total_pos > 0 else 1
        total_neg = total_neg if total_neg > 0 else 1

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                try:
                    series_binned = pd.qcut(df[col], q=self.bins, duplicates='drop').astype(str)
                except ValueError:
                    series_binned = pd.cut(df[col], bins=self.bins).astype(str)
            else:
                series_binned = df[col].astype(str)

            stats = pd.DataFrame({'feature': series_binned, 'target': y_arr})
            group_counts = stats.groupby('feature')['target'].agg(pos='sum', total='count')
            group_counts['neg'] = group_counts['total'] - group_counts['pos']

            group_counts['pos_dist'] = (group_counts['pos'] + 0.5) / total_pos
            group_counts['neg_dist'] = (group_counts['neg'] + 0.5) / total_neg

            group_counts['woe'] = np.log(group_counts['pos_dist'] / group_counts['neg_dist'])
            group_counts['iv'] = (group_counts['pos_dist'] - group_counts['neg_dist']) * group_counts['woe']

            self.woe_maps_[col] = group_counts['woe'].to_dict()
            self.iv_scores_[col] = group_counts['iv'].sum()

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        df_woe = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                try:
                    series_binned = pd.qcut(df[col], q=self.bins, duplicates='drop').astype(str)
                except ValueError:
                    series_binned = pd.cut(df[col], bins=self.bins).astype(str)
            else:
                series_binned = df[col].astype(str)

            df_woe[f'{col}_woe'] = series_binned.map(self.woe_maps_[col]).fillna(0)
            
        return df_woe


def build_feature_engineering_pipeline() -> Pipeline:
    numeric_features = ['Amount', 'Value', 'Total_Amount_Cust', 'Avg_Amount_Cust', 'Transaction_Count_Cust', 'Std_Amount_Cust']
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId', 'ProductId', 'CurrencyCode']
    temporal_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features),
            ('time', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('scaler', StandardScaler())]), temporal_features)
        ],
        remainder='passthrough'
    )

    return Pipeline([
        ('feature_engineer', XenteFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('woe_transform', NativeWoETransformer(bins=5))
    ])

