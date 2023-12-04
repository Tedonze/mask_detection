from pandas import read_csv
from .core.configs import settings, fs


def splitted_dataframes(fracs: list):
    # assert (len(fracs) == 3) & (sum(fracs) == 1), "Fracs inconvenient"
    df_classes = read_csv(
        fs.open(f"{settings.s3_prefix}_classes.csv")
    ).drop(axis=1, columns=[" without_mask"])
    df_train = (
        df_classes
        .groupby(' with_mask', group_keys=False)
        .apply(lambda x: x.sample(frac=fracs[0]))
    )
    df_validation_test = df_classes[~df_classes.index.isin(df_train.index)]
    df_validation = (
        df_validation_test
        .groupby(' with_mask', group_keys=False)
        .apply(lambda x: x.sample(frac=fracs[1]/(1-fracs[0])))
    )
    df_test = df_validation_test[
        ~df_validation_test.index.isin(df_validation.index)]

    return df_train, df_validation, df_test
