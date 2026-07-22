import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED


def build_features():
    # -----------------------------
    # Load datasets
    # -----------------------------
    train = pd.read_csv(
        DATA_RAW / 'train.csv',
        dtype={'StateHoliday': 'string'},
        low_memory=False
    )

    store = pd.read_csv(DATA_RAW / 'store.csv')

    # -----------------------------
    # Merge train + store
    # -----------------------------
    df = train.merge(store, on='Store', how='left')

    # -----------------------------
    # Date handling
    # -----------------------------
    df['Date'] = pd.to_datetime(df['Date'])

    # -----------------------------
    # Keep only open stores with sales
    # -----------------------------
    df = df[(df['Open'] == 1) & (df['Sales'] > 0)]

    # -----------------------------
    # Sort for time-series operations
    # -----------------------------
    df = df.sort_values(['Store', 'Date'])

    # -----------------------------
    # Calendar features
    # -----------------------------
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['DayOfWeek'] >= 6).astype(int)

    # -----------------------------
    # Lag features
    # -----------------------------
    df['sales_lag_1'] = df.groupby('Store')['Sales'].shift(1)
    df['sales_lag_7'] = df.groupby('Store')['Sales'].shift(7)

    # -----------------------------
    # Rolling features (safe version)
    # -----------------------------
    df['rolling_mean_7'] = (
        df.groupby('Store')['Sales']
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    df['rolling_std_7'] = (
        df.groupby('Store')['Sales']
        .transform(lambda x: x.shift(1).rolling(7).std())
    )

    # -----------------------------
    # Remove rows with missing lag history
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # Save processed dataset
    # -----------------------------
    output_path = DATA_PROCESSED / 'retail_features.parquet'

    df.to_parquet(
        output_path,
        engine='pyarrow',
        index=False
    )

    print(f'Processed dataset saved to: {output_path}')
    print(f'Final shape: {df.shape}')

    return df


if __name__ == '__main__':
    build_features()