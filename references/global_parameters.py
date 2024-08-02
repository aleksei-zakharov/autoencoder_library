# because we run ipynb from notebooks/vol/ or notebooks/mnist folders, we need to go 2 levels up
GLOBAL_DATA_PATH = '../../data/'  


READ_RAW_DATA_FILE_PATH = GLOBAL_DATA_PATH + 'raw/TR_Data.xlsx'
READ_RAW_DATA_FILE_SHEET_NAME = 'DownloadedDataNoFormula'
SAVE_ATM_VOLS_FILE_PATH = GLOBAL_DATA_PATH + 'processed/atm_vol.xlsx'
SAVE_ATM_VOLS_FILE_SHEET_NAME = 'atm_vol'
SAVE_VOL_CUBE_FILE_PATH = GLOBAL_DATA_PATH + 'processed/vol_cube.xlsx'
SAVE_VOL_CUBE_FILE_SHEET_NAME = 'vol_cube'