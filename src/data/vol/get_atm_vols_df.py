import pandas as pd

from references.global_parameters import SAVE_ATM_VOLS_FILE_PATH, \
                                         SAVE_ATM_VOLS_FILE_SHEET_NAME


def get_atm_vols_df():    
    df = pd.read_excel(SAVE_ATM_VOLS_FILE_PATH, 
                       sheet_name=SAVE_ATM_VOLS_FILE_SHEET_NAME,
                       index_col=0)
    return df