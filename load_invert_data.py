import numpy as np
from pathlib import Path
import h5py
import pandas as pd


BUILDING_CLASS_COLUMNS = {
    "index": int,
    "name": str,
    "construction_period_start": int,
    "construction_period_end": int,
    "building_categories_index": int,
    "number_of_dwellings_per_building": "float32",
    "number_of_persons_per_dwelling": "float32",
    "length_of_building": "float32",
    "width_of_building": "float32",
    "number_of_floors": "float32",
    "room_height": "float32",
    "percentage_of_building_surface_attached_length": "float32",
    "percentage_of_building_surface_attached_width": "float32",
    "share_of_window_area_on_gross_surface_area": "float32",
    "share_of_windows_oriented_to_south": "float32",
    "share_of_windows_oriented_to_north": "float32",
    "grossfloor_area": "float32",
    "heated_area": "float32",
    "areafloor": "float32",
    "areawindows": "float32",
    "area_suitable_solar": "float32",
    "grossvolume": "float32",
    "heatedvolume": "float32",
    "heated_norm_volume": "float32",
    "hwb": "float32",
    "hwb_norm": "float32",
    "u_value_ceiling": "float32",
    "u_value_exterior_walls": "float32",
    "u_value_windows1": "float32",
    "u_value_windows2": "float32",
    "u_value_roof": "float32",
    "u_value_floor": "float32",
    "seam_loss_windows": "float32",
    "trans_loss_walls": "float32",
    "trans_loss_ceil": "float32",
    "trans_loss_wind": "float32",
    "trans_loss_floor": "float32",
    "trans_loss_therm_bridge": "float32",
    "trans_loss_ventilation": "float32",
    "total_heat_losses": "float32",
    "average_effective_area_wind_west_east_red_cool": "float32",
    "average_effective_area_wind_south_red_cool": "float32",
    "average_effective_area_wind_north_red_cool": "float32",
    "spec_int_gains_cool_watt": "float32",
    "attached_surface_area": "float32"
}

BUILDING_SEGMENT_COLUMNS = {
    "index": int,
    "name": str,
    "building_classes_index": int,
    "number_of_buildings": "float32",
    "heat_supply_system_index": int,
    "installation_year_system_start": "float32",
    "installation_year_system_end": "float32",
    "distribution_sh_index": int,
    "distribution_dhw_index": int,
    "pv_system_index": int,
    "energy_carrier": int,
    "annual_energy_costs_hs": "float32",
    "total_annual_cost_hs": "float32",
    "annual_energy_costs_dhw": "float32",
    "total_annual_cost_dhw": "float32",
    "hs_efficiency": "float32",
    "dhw_efficiency": "float32",
    "size_pv_system": "float32",
    "fed_ambient_sh_per_bssh": "float32",
    "fed_ambient_dhw_per_bssh": "float32",
}

HEATING_SYSTEM_INDEX = {
    1: "no heating",
    2: "no heating",
    3: "district heating",
    4: "district heating",
    5: "district heating",
    6: "district heating",
    7: "district heating",
    8: "district heating",
    9: "oil",
    10: "oil",
    11: "oil",
    12: "oil",
    13: "oil",
    14: "oil",
    15: "oil",
    16: "oil",
    17: "oil",
    18: "coal",
    19: "coal",
    20: "coal",
    21: "gas",
    22: "gas",
    23: "gas",
    24: "gas",
    25: "gas",
    26: "gas",
    27: "gas",
    28: "gas",
    29: "wood",
    30: "wood",
    31: "wood",
    32: "wood",
    33: "wood",
    34: "wood",
    35: "wood",
    36: "wood",
    37: "electricity",  # TODO rein nehmen
    38: "electricity",  # TODO rein nehmen
    39: "electricity",  # TODO rein nehmen
    40: "split system",  # TODO rein nehmen!
    41: "split system",  # TODO rein nehmen!
    42: "heat pump air",
    43: "heat pump ground",
    44: "electricity"  # TODO rein nehmen
}

def select_invert_building(gdf_row, invert_selection: pd.DataFrame):
    # check if the construction year from urban3r is available in invert:

    if gdf_row['invert_construction_period'] in invert_selection["construction_period"]:
        invert_selection.query(f"construction_period == {gdf_row['invert_construction_period']}")

    # if it is not available take the closest
    else:
        pass


def hdf5_to_pandas(hdf5_file: Path, group_name, columns) -> pd.DataFrame:
    with h5py.File(hdf5_file, 'r') as file:
        # Get the table from the group
        dataset = file[group_name]
        df = pd.DataFrame(index=range(len(dataset)), columns=[list(columns.keys())])
        for name in columns.keys():
            df[name] = dataset[name]

    return df


def get_number_energy_carriers_from_invert(group: pd.DataFrame) -> dict:
    numbers = group.groupby("energy_carrier_name")["number_of_buildings"].sum()
    return numbers.to_dict()


def to_series(col):
    if isinstance(col, (pd.DataFrame, pd.Series)):
        return col.squeeze()
    elif isinstance(col, (list, np.ndarray)):
        return pd.Series(col)
    else:
        return col


def calc_mean(data: dict) -> float:
    # Multiply each key with its corresponding value and add them
    sum_products = sum(key * value for key, value in data.items())
    # Calculate the sum of the values
    sum_values = sum(value for value in data.values())
    # Return the mean value
    if sum_values == 0:
        return np.nan
    return sum_products / sum_values


def calculate_mean_supply_temperature(grouped_df: pd.DataFrame,
                                      heating_system_name: str = None,
                                      helper_name: str = None) -> float:
    # add supply temperature to bc df:
    # check if there are multiple supply temperatures:
    supply_temperatures = list(grouped_df.loc[:, "supply_temperature"].unique())
    if len(supply_temperatures) > 1:
        # group by supply temperature
        supply_temperature_group = grouped_df.groupby("supply_temperature")
        nums = {}
        for temp in supply_temperatures:
            if helper_name == "get_number_of_buildings":
                number_buildings_sup_temp = supply_temperature_group.get_group(temp)["number_of_buildings"].sum()
            else:
                number_buildings_sup_temp = supply_temperature_group.get_group(temp)[heating_system_name].sum()

            nums[temp] = number_buildings_sup_temp
        # calculate the mean:
        # sometimes the supply temperature is 1000 °C because
        mean_sup_temp = calc_mean(nums)
    else:
        mean_sup_temp = supply_temperatures[0]

    return mean_sup_temp


def calculate_mean(grouped: pd.DataFrame, names: list, heating_system_name: str):
    if grouped[heating_system_name].sum() == 0:  # create nan row so it can be easily dropped later
        new_row = pd.Series(data=[np.nan] * len(grouped[names].columns), index=grouped[names].columns)
    else:
        weights = grouped[heating_system_name] / grouped[heating_system_name].sum()
        new_row = (grouped[names].T * weights).T.sum()
    return new_row


def create_representative_building(group: pd.DataFrame,
                                   column_name_with_numbers: str,
                                   merging_names: list,
                                   adding_names: list) -> pd.DataFrame:
    new_row = pd.DataFrame(columns=group.columns, index=[0])
    # representative air source HP building
    new_row.loc[0, merging_names] = calculate_mean(group, names=merging_names, heating_system_name=column_name_with_numbers)
    new_row.loc[0, adding_names] = group.loc[:, adding_names].sum()

    new_row.loc[0, "supply_temperature"] = calculate_mean_supply_temperature(
        group,
        heating_system_name=column_name_with_numbers
    )
    # new name is first 5 letters of first name + heating system
    heating_system = column_name_with_numbers.split("_")[-1]
    new_row["construction_period_start"] = group["construction_period_start"].unique()
    new_row["construction_period_end"] = group["construction_period_end"].unique()[0]
    new_row.loc[0, "name"] = f"{str(group.loc[:, 'name'].iloc[0][0:5])[2:]}_{heating_system}"
    new_row.loc[0, "index"] = group.loc[:, "index"].iloc[0]  # new index is the first index of merged rows
    return new_row


def get_number_of_buildings_from_invert() -> pd.DataFrame:
    hdf5_f = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\001_buildings_spain.hdf5")
    bc_df = hdf5_to_pandas(hdf5_f, f"BC_{2020}", BUILDING_CLASS_COLUMNS)
    bssh_df = hdf5_to_pandas(hdf5_f, f"BSSH_{2020}", BUILDING_SEGMENT_COLUMNS)
    # reomve multiindex
    bssh_df.columns = bssh_df.columns.map(''.join)
    bc_df.columns = bc_df.columns.map(''.join)
    # change columns to series:
    bssh = bssh_df.apply(to_series)
    bc = bc_df.apply(to_series)
    bc["building_categories_index"] = bc["building_categories_index"].astype(int)

    # Group the rows of bssh_df by building_classes_index
    bssh_grouped = bssh.groupby(["building_classes_index", "heat_supply_system_index"])

    for (index, heat_system), group in bssh_grouped:
        # add total number of buildings:
        total_number_buildings = group["number_of_buildings"].sum()
        bc.loc[bc.loc[:, "index"] == index, "number_of_buildings"] = total_number_buildings
        bc.loc[bc.loc[:, "index"] == index, "heating_medium"] = HEATING_SYSTEM_INDEX[heat_system]

    # # columns where numbers are summed up (PV and number of buildings)
    # adding_names = [name for name in bc.columns if "number" in name]
    # # columns to merge: [2:] so index and name are left out
    # merging_names = [
    #                     name for name in bc.columns if "PV" and "number" not in name and "construction" not in name
    #                 ][2:] + adding_names[:3]
    # # except number of persons and number of dwellings ([3:]) left out
    # adding_names = adding_names[3:]
    #
    # # group the heating mediums together
    # final_df = pd.DataFrame()
    # bc_group = bc.groupby("heating_medium")
    # for heating_medium, group in bc_group:
    #     new_building = create_representative_building(group=group,
    #                                                   column_name_with_numbers="number_of_buildings",
    #                                                   merging_names=merging_names,
    #                                                   adding_names=adding_names)

    return bc


if __name__ == "__main__":
    get_number_of_buildings_from_invert()