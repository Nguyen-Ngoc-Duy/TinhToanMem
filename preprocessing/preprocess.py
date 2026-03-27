import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def describe_data(df):
    print("===== MÔ TẢ DỮ LIỆU =====")
    print("Số mẫu:", df.shape[0])
    print("Số đặc trưng:", df.shape[1])

    print("\nKiểu dữ liệu:")
    print(df.dtypes)

    print("\nSố lượng missing:")
    print(df.isnull().sum())


def preprocess_data(df):
    df_clean = df.copy()

    df_clean['seller_type'] = df_clean['seller_type'].map({
        'Individual': 1,
        'Dealer': 0
    })

    df_clean['owner'] = df_clean['owner'].str.extract(r'(\d+)').astype(float)

    df_clean['age'] = 2026 - df_clean['year']
    df_clean = df_clean.drop(columns=['year'])
    df_clean = df_clean.dropna(subset=['ex_showroom_price'])
    df_clean = df_clean.dropna()

    print("\n===== SAU XỬ LÝ =====")
    print("Số mẫu:", df_clean.shape[0])
    print("Số đặc trưng:", df_clean.shape[1])

    print("\nMissing còn lại:")
    print(df_clean.isnull().sum())

    return df_clean