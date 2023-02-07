def dataset_info(df):
    df.info()
    print(df.describe().T)
    print(df.isnull().sum())

def replacing_null_values_in_dataset(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    print("Null values checking: {}.".
        format("null values found" if df.isnull().sum().sum() > 0  else "null values not found"))
    return df 

def preparing_dataset(df):
    df = replacing_null_values_in_dataset(df)
    df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
    df.replace({'white': 1, 'red': 0}, inplace=True)
    return df
