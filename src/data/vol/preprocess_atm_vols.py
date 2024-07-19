import pandas as pd

from references.global_parameters import READ_RAW_DATA_FILE_PATH, \
                                         READ_RAW_DATA_FILE_SHEET_NAME, \
                                         SAVE_ATM_VOLS_FILE_PATH, \
                                         SAVE_ATM_VOLS_FILE_SHEET_NAME


def preprocess_atm_vols():

    def tenor_tenor(name):
        sep_idx = name.find('X')
        atm_idx = name.find('ATM')
        return name[3:sep_idx] + '_' + name[sep_idx + 1:atm_idx]

    df = pd.read_excel(READ_RAW_DATA_FILE_PATH, sheet_name=READ_RAW_DATA_FILE_SHEET_NAME)
    df = df.loc[1:,:]  # filter the first row without numbers

    # Dataframe with dates as indexes, different option/swap tenors as columns and vols as data
    df_vols = pd.DataFrame()  

    n_cols = len(df.columns)
    for i in range(n_cols // 2):
        date_col_num = 2 * i
        vol_col_num = 2 * i + 1
        col_name = df.columns[2 * i]

        if col_name.find('ATM') != - 1:  # if column name contains 'ATM'
            # Add 1 column to df_vols dataframe
            data = df.iloc[:,vol_col_num]
            index = df.iloc[:,date_col_num].infer_objects()
            df_vol = pd.DataFrame(data=list(data),
                                index=index,
                                columns=[col_name])
            df_vol.dropna(axis='index', how='any', inplace=True)
            df_vols = pd.concat([df_vols, df_vol], join ='outer', axis=1)

    # Drop NaN data
    df_vols.dropna(axis='index', how='any', inplace=True)

    # Rename the columns
    df_vols.columns = [tenor_tenor(x) for x in df_vols.columns]
    # df = df.reindex(sorted(df.columns), axis=1)  # sorting of columns

    # Export to the file
    df_vols.to_excel(SAVE_ATM_VOLS_FILE_PATH, sheet_name=SAVE_ATM_VOLS_FILE_SHEET_NAME)


if __name__ == "__main__":
    preprocessing_atm()
    pass