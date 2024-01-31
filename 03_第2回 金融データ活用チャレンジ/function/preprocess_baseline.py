import pandas as pd
import numpy as np

def preprocess(df, replace_dict=None, ce_dict=None):
    # 貸借手の所在地系の変数
    # City: Cityは汎用性が低いと考えられるためDrop
    df.drop("City", axis=1, inplace=True)

    # 借り手の会社に関する変数（Sector, FranchiseCode）
    # 31-33, 44-45, 48-49 は同じらしい => 32,33を31に, 45を44に, 49を48に変換
    code_dict = {
        32: 31,
        33: 31,
        45: 44,
        49: 48
    }
    df["Sector"] = df["Sector"].replace(code_dict)

    # 今回の借り入れに関する変数（RevLineCr, LowDoc）
    # 公式ページには値の候補が2つ（YesとNoのYN）と記載があるが、実際の値の種類は2より多い。YN以外はNaNへ置換
    revline_dict = {'0': np.nan, 'T': np.nan}
    df["RevLineCr"] = df["RevLineCr"].replace(revline_dict)

    lowdoc_dict = {'C': np.nan, '0': np.nan, 'S': np.nan, 'A': np.nan}
    df["LowDoc"] = df["LowDoc"].replace(lowdoc_dict)

    # 日付系の変数（DisbursementDate, ApprovalDate）
    # 日付型へ変更 → 年を抽出（借りた月や日にはあまり意味はないと思われるため）
    df['DisbursementDate'] = pd.to_datetime(df['DisbursementDate'], format='%d-%b-%y')
    df["DisbursementYear"] = df["DisbursementDate"].dt.year
    df.drop(["DisbursementDate", "ApprovalDate"], axis=1, inplace=True)

    # 本来数値型のものを変換する
    cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    df[cols] = df[cols].applymap(lambda x: x.strip().replace('$', '').replace(',', '')).astype(float).astype(int)

    # 特徴量エンジニアリング
    df["FY_Diff"] = df["ApprovalFY"] - df["DisbursementYear"]
    df["State_is_BankState"] = (df["State"] == df["BankState"])
    df["State_is_BankState"] = df["State_is_BankState"].replace({True: 1, False: 0})

    df['SBA_Portion'] = df['SBA_Appv'] / df['GrAppv']
    df["DisbursementGrossRatio"] = df["DisbursementGross"] / df["GrAppv"]
    df["MonthlyRepayment"] = df["GrAppv"] / df["Term"]
    df["NullCount"] = df.isnull().sum(axis=1)

    # カテゴリカル変数の設定
    df[cols_category] = df[cols_category].fillna(-1)

    # train
    if replace_dict is None:
        # countencode, labelencode
        # ce_dict: 列名を入れるとそのカテゴリのデータがどのくらいあるかを返してくれます
        # replace_dict: 列名を入れるとlabelencodeのための数字を返してくれます
        ce_dict = {}
        replace_dict = {}
        for col in cols_category:
            replace_dict[col] = {}
            vc = df[col].value_counts()
            ce_dict[col] = vc
            replace_dict_in_dict = {}
            for i, k in enumerate(vc.keys()):
                replace_dict_in_dict[k] = i
            replace_dict[col] = replace_dict_in_dict
            df[f"{col}_CountEncode"] = df[col].replace(vc).astype(int)
            df[col] = df[col].replace(replace_dict_in_dict).astype(int)
        return df, replace_dict, ce_dict

    # test
    else:
        for col in cols_category:
            # カウントエンコード
            test_vals_uniq = df[col].unique()
            ce_dict_in_dict = ce_dict[col]
            for test_val in test_vals_uniq:
                if test_val not in ce_dict_in_dict.keys():
                    ce_dict_in_dict[test_val] = -1
            df[f"{col}_CountEncode"] = df[col].replace(ce_dict_in_dict).astype(int)

            # LabelEncode
            test_vals_uniq = df[col].unique()
            replace_dict_in_dict = replace_dict[col]
            for test_val in test_vals_uniq:
                if test_val not in replace_dict_in_dict.keys():
                    replace_dict_in_dict[test_val] = -1
            df[col] = df[col].replace(replace_dict_in_dict).astype(int)
        return df