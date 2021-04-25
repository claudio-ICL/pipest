from pipest.impact.bacrymuzy.main import measure_impact, panmeasure, collect_results

symbol="INTC"
date="2019-01-23"
time_window="41400-45000"
initial_inventory=10.0
liquidator_base_rate=0.150
liquidator_control=0.2

def simple(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    enforce_price_symmetry=False,
    type_of_liquid='constant_intensity',
    initial_inventory=10.0,
    liquidator_base_rate=0.150,
    liquidator_control=0.2,
):
    return measure_impact(
        symbol=symbol,
        date=date,
        time_window=time_window,
        enforce_price_symmetry=enforce_price_symmetry,
        type_of_liquid=type_of_liquid,
        initial_inventory=initial_inventory,
        liquidator_base_rate=liquidator_base_rate,
        liquidator_control=liquidator_control,
        liquidator_control_type='fraction_of_bid_side',
        dump=False,
        return_=True)

def main(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    initial_inventory=10.0,
    liquidator_base_rate=0.150,
    liquidator_control=0.2,
):
    panmeasure(
        symbol=symbol,
        date=date,
        time_window=time_window,
        initial_inventory=initial_inventory,
        liquidator_base_rate=liquidator_base_rate,
        liquidator_control=liquidator_control,
        use_pbs_array_index=False,
    )
    collect_results(
        symbol=symbol,
        date=date,
        time_window=time_window,
    )
