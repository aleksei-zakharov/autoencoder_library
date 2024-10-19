# import numpy as np
from numpy import unique, zeros

from src.data.vol.get_vol_cube_df import get_vol_cube_df


def get_vol_cube_tenors_strikes_dates():
    """
    Based on dataframe from get_vol_cube_df function,
    returns volatility cubes, option tenors, swap tenors, strikes and dates

    Returns:

        vol_cube: for each date, we have volatility cube 3D structure that consists
                of volatility values (in bp) for different option tenors, swap tenors and strikes

        uniq_opt_tenors: list of possible option tenors (second dimension of vol_cube structure)

        uniq_swap_tenors: list of possible swap tenors  (third dimension of vol_cube structure)

        uniq_strikes: list of possible strikes (fourth dimension of vol_cube structure)

        dates: list of dates (first dimension of vol_cube structure)

    """

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
        if len(skew) == 3:      # skew='ATM'
            return 0
        elif skew[3] == '-':    # e.g. skew='ATM-50bp'
            return -int(skew[4:-2])
        else:                   # e.g. skew='ATM+50bp'
            return int(skew[4:-2])

    df = get_vol_cube_df()

    # Get unique tenors and skews out of all dataframe columns
    opt_tenors = [i[0] for i in df.columns.str.split('_')]
    uniq_opt_tenors = unique(opt_tenors)

    swap_tenors = [i[1] for i in df.columns.str.split('_')]
    uniq_swap_tenors = unique(swap_tenors)

    strikes = [i[2] for i in df.columns.str.split('_')]
    uniq_strikes = unique(strikes)

    # Sort tenors and skews
    uniq_opt_tenors = sorted(uniq_opt_tenors, key=days_in_label)
    uniq_swap_tenors = sorted(uniq_swap_tenors, key=days_in_label)
    uniq_strikes = sorted(uniq_strikes, key=bp_in_skew)

    # Create vol cube
    vol_cube = zeros([len(df), len(uniq_opt_tenors), len(uniq_swap_tenors), len(uniq_strikes)])

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
              