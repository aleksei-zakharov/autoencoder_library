from pandas import read_excel

from references.global_parameters import SAVE_VOL_CUBE_FILE_PATH, \
                                         SAVE_VOL_CUBE_FILE_SHEET_NAME


def get_vol_cube_df():
    """
    Download (from excel file ) DataFrame with volatilities for different strikes, 
    opt and swap tenors and dates 
    """
    
    df = read_excel(SAVE_VOL_CUBE_FILE_PATH, 
                       sheet_name=SAVE_VOL_CUBE_FILE_SHEET_NAME,
                       index_col=0)
    return df