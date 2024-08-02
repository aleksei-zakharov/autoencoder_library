import numpy as np

from src.data.vol.get_vol_cube_df import get_vol_cube_df


def get_vol_cube_tenors_strikes_dates():

    def days_in_label(s):
        if s[-1] == 'Y':
            return int(s[:-1])*365
        elif s[-1] == 'M':
            return int(s[:-1])*30
        elif s[-1] == 'D':
            return int(s[:-1])
        else:
            return 'Error'
        
    def bp_in_skew(skew):
        if skew[0] == 'A':
            return 0
        elif skew[0] == 'N':
            return -int(skew[1:-2])
        else:
            return int(skew[1:-2])

    df = get_vol_cube_df()

    # Get unique tenors and skews
    opt_tenors = [i[0] for i in df.columns.str.split('_')]
    uniq_opt_tenors = np.unique(opt_tenors)

    swap_tenors = [i[1] for i in df.columns.str.split('_')]
    uniq_swap_tenors = np.unique(swap_tenors)

    strikes = [i[2] for i in df.columns.str.split('_')]
    uniq_strikes = np.unique(strikes)

    # Sort tenors and skews
    uniq_opt_tenors = sorted(uniq_opt_tenors, key=days_in_label)
    uniq_swap_tenors = sorted(uniq_swap_tenors, key=days_in_label)
    uniq_strikes = sorted(uniq_strikes, key=bp_in_skew)

    # Create vol cube
    vol_cube = np.zeros([len(df), len(uniq_opt_tenors), len(uniq_swap_tenors), len(uniq_strikes)])

    for i in range(len(df)):
        for j, val in enumerate(df.columns):
            opt_tenor, swap_tenor, strike = val.split('_')
                    
            idx1 = uniq_opt_tenors.index(opt_tenor)
            idx2 = uniq_swap_tenors.index(swap_tenor)
            idx3 = uniq_strikes.index(strike)
            
            vol_cube[i, idx1, idx2, idx3] = df.iloc[i, j]

    # Get dates out of dataframe indexes
    dates = [i.date() for i in df.index]

    return vol_cube, \
           uniq_opt_tenors, \
           uniq_swap_tenors, \
           uniq_strikes, \
           dates
              