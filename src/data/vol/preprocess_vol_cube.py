import pandas as pd

from references.global_parameters import READ_RAW_DATA_FILE_PATH, \
                                         READ_RAW_DATA_FILE_SHEET_NAME, \
                                         SAVE_VOL_CUBE_FILE_PATH, \
                                         SAVE_VOL_CUBE_FILE_SHEET_NAME


def preprocess_vol_cube():

    def tenor_tenor_skew(name):
        # create tenor_tenor_skew string like '3M_1Y_P50bp' or '1Y_10Y_ATM'
        sep_idx = name.find('X')
        skew_idx = max(name.find('A'), name.find('N'), name.find('P'))  # name looks like ...ATM=R or ...N1=R or ....P4=R

        match name[skew_idx + 1]:
            case '1':
                shift_bp = '25bp'
            case '2':
                shift_bp = '50bp'
            case '3':
                shift_bp = '100bp'

        if name[skew_idx] == 'A':
            skew_str = 'ATM'
        elif name[skew_idx] == 'P':
            skew_str = 'P' + shift_bp
        else:
            skew_str = 'N' + shift_bp
        
        return name[3:sep_idx] + '_' + name[sep_idx + 1 : skew_idx] + '_' + skew_str


    df = pd.read_excel(READ_RAW_DATA_FILE_PATH, sheet_name=READ_RAW_DATA_FILE_SHEET_NAME)
    df = df.loc[1:,:]  # filter the first row without numbers

    # Dataframe with dates as indexes, different option/swap tenors as columns and vols as data
    df_vols = pd.DataFrame()  

    n_cols = len(df.columns)
    for i in range(n_cols // 2):
        date_col_num = 2 * i
        vol_col_num = 2 * i + 1
        col_name = df.columns[2 * i]

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

    #  Drop all rows where there is at least one zero value in a row
    df_vols = df_vols.loc[(df_vols!=0).all(axis=1)]

    # Rename the columns
    df_vols.columns = [tenor_tenor_skew(x) for x in df_vols.columns]
    # df = df.reindex(sorted(df.columns), axis=1)  # sorting of columns

    # Export to the file
    df_vols.to_excel(SAVE_VOL_CUBE_FILE_PATH, sheet_name=SAVE_VOL_CUBE_FILE_SHEET_NAME)