import numpy as np
from pipest.lobster.py_scripts import prepare_from_lobster as from_lobster

def main(
     LOB=None,
     mf=None,
     symbol='INTC',
     date='2019-01-31',
     ticksize=100,
     n_levels=2,
):
    man_mf = from_lobster.ManipulateMessageFile(
                 LOB=LOB, 
                 mf=mf,
                 symbol=symbol,
                 date=date,
                 ticksize=ticksize,
                 n_levels=n_levels,
                 )
    man_ob = from_lobster.ManipulateOrderBook(LOB=man_mf.LOB, 
            symbol = man_mf.symbol,
            date = man_mf.date,
            n_levels = man_mf.n_levels,
            list_of_n_states = [3,3],
            volume_imbalance_upto_level=2
            )
    man_ob.set_states(midprice_changes = np.array(man_mf.messagefile['sign_delta_mid'].values, dtype=np.int))
    time_origin = man_mf.messagefile['time'].min()
    data = from_lobster.DataToStore(man_ob, man_mf, time_origin=time_origin)
    return data





