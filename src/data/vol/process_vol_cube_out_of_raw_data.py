from pandas import read_excel, DataFrame, concat
from datetime import date

from references.global_parameters import READ_RAW_DATA_FILE_PATH, \
                                         READ_RAW_DATA_FILE_SHEET_NAME, \
                                         SAVE_VOL_CUBE_FILE_PATH, \
                                         SAVE_VOL_CUBE_FILE_SHEET_NAME


def process_vol_cube_out_of_raw_data():
    """
    Download vol data from "data\raw" folder, create a useful dataframe with it 
    and save it to Excel file in "data\preprocessed" folder
    
    Methods:

        find_skew_idx: find the index in a string where the information about skew starts

        tenor_tenor_skew: creates a new column name in format "tenor_tenor_skew" out of initial column name
        
        skew_number: if skew is not "ATM", provide the number that goes right after "P" or "N" letter 
            in a skew name. For instance, "P50" means that the strike is "ATM + 50 basis points" and 
            "N25" means that the strike is "ATM - 25 basis points"
    """


    def find_skew_idx(name):
        return max(name.find('A'), name.find('N'), name.find('P'))  # name looks like ...ATM=R or ...N1=R or ....P4=R


    def tenor_tenor_skew(name):
        # create tenor_tenor_skew string like '3M_1Y_ATM-50bp' or '1Y_10Y_ATM' 
        # from like 'USD10YX7YN3=R' or 'USD1MX1YATM=R'
        sep_idx = name.find('X')
        skew_idx = find_skew_idx(name)

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
            skew_str = 'ATM+' + shift_bp
        else:
            skew_str = 'ATM-' + shift_bp
        
        return name[3:sep_idx] + '_' + name[sep_idx + 1 : skew_idx] + '_' + skew_str
    

    def skew_number(name):
        skew_idx = find_skew_idx(name)
        return name[skew_idx + 1]


    df = read_excel(READ_RAW_DATA_FILE_PATH, sheet_name=READ_RAW_DATA_FILE_SHEET_NAME)
    df = df.loc[1:,:]  # filter the first row without numbers

    # Dataframe with dates as indexes, different option/swap tenors as columns and vols as data
    df_vols = DataFrame()  

    n_cols = len(df.columns)
    for i in range(n_cols // 2):
        date_col_num = 2 * i
        vol_col_num = 2 * i + 1
        col_name = df.columns[2 * i]

        # Take only ATM, ATM+50bp, ATM+100bp out of ['T', '1', '2', '3', '4', '5']
        if skew_number(col_name) in ['T', '2', '3']:  

            # Add 1 column to df_vols dataframe
            data = df.iloc[:,vol_col_num]
            index = df.iloc[:,date_col_num].infer_objects()
            df_vol = DataFrame(data=list(data),
                                index=index,
                                columns=[col_name])
            df_vol.dropna(axis='index', how='any', inplace=True)
            df_vols = concat([df_vols, df_vol], join ='outer', axis=1)

    # Drop NaN data
    df_vols.dropna(axis='index', how='any', inplace=True)

    #  Drop all rows where there is at least one zero value in a row
    df_vols = df_vols.loc[(df_vols!=0).all(axis=1)]

    # Rename the columns
    df_vols.columns = [tenor_tenor_skew(x) for x in df_vols.columns]

    # Get rid of first dates where vols are outliers
    OUTLIERS = [date(2022, 2, 4), date(2022, 2, 7), date(2022, 2, 8), date(2022, 2, 9)]
    df_vols = df_vols[~df_vols.index.isin(OUTLIERS)]

    # Export to the file
    df_vols.to_excel(SAVE_VOL_CUBE_FILE_PATH, sheet_name=SAVE_VOL_CUBE_FILE_SHEET_NAME)