import numpy as np

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

def five_fold_split(date_frame, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(date_frame.index)
    m = len(date_frame.index)

    block1_end = int(0.2 * m)
    block2_end = int(0.2 * m + block1_end)
    block3_end = int(0.2 * m + block2_end)
    block4_end = int(0.2 * m + block3_end)
    block5_end = int(0.2 * m + block4_end)

    block1 = date_frame.iloc[perm[:block1_end]]
    block2 = date_frame.iloc[perm[block1_end: block2_end]]
    block3 = date_frame.iloc[perm[block2_end: block3_end]]
    block4 = date_frame.iloc[perm[block3_end: block4_end]]
    block5 = date_frame.iloc[perm[block4_end:]]

    blocks = []

    blocks.append(block1)
    blocks.append(block2)
    blocks.append(block3)
    blocks.append(block4)
    blocks.append(block5)

    return blocks

def train_validate_cross(blocks, k):
    train = pd.DataFrame()
    validate = pd.DataFrame()

    for i, block in enumerate(blocks):
        if k != i:
            train = train.append(block)
        else:
            validate = validate.append(block)   
    return train, validate