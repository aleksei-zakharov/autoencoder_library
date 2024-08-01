import numpy as np

from src.data.vol.get_atm_vols import get_atm_vols

def get_atm_vol_surface():


    def days_in_label(s):
        if s[-1] == 'Y':
            return int(s[:-1])*365
        elif s[-1] == 'M':
            return int(s[:-1])*30
        elif s[-1] == 'D':
            return int(s[:-1])
        else:
            return 'Error'


    df = get_atm_vols()

    opt_tenors = [i[0] for i in df.columns.str.split('_')]
    uniq_opt_tenors = np.unique(opt_tenors)

    swap_tenors = [i[1] for i in df.columns.str.split('_')]
    uniq_swap_tenors = np.unique(swap_tenors)

    # sort
    uniq_opt_tenors = sorted(uniq_opt_tenors, key=days_in_label)
    uniq_swap_tenors = sorted(uniq_swap_tenors, key=days_in_label)

    atm_data_surf = np.zeros([len(df), len(uniq_opt_tenors), len(uniq_swap_tenors)])

    for i in range(len(df)):
        for j, val in enumerate(df.columns):
            opt_tenor, swap_tenor = val.split('_')
                    
            idx1 = uniq_opt_tenors.index(opt_tenor)
            idx2 = uniq_swap_tenors.index(swap_tenor)
            
            atm_data_surf[i, idx1, idx2] = df.iloc[i, j]

    dates = [i.date() for i in df.index]

    return atm_data_surf, \
           uniq_opt_tenors, \
           uniq_swap_tenors, \
           dates
              