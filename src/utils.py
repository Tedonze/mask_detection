from pandas import read_csv
from .core.configs import settings, fs


def splitted_dataframes(fracs: float):
    # assert (len(fracs) == 3) & (sum(fracs) == 1), "Fracs inconvenient"
    assert(0 < fracs <1)
    df_classes = read_csv(
        fs.open(f"{settings.s3_prefix}_classes.csv")
    ).drop(axis=1, columns=[" without_mask"])
    df_train = (
        df_classes
        .groupby(' with_mask', group_keys=False)
        .apply(lambda x: x.sample(frac=fracs))
    )
    df_validation = df_classes[~df_classes.index.isin(df_train.index)]

    
    return df_train, df_validation
