import numpy as np
from pathlib import Path
import h5py
import pandas as pd
import tqdm

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
    "attached_surface_area": "float32",
    "n_50": "float32",
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
    # "annual_energy_costs_hs": "float32",
    # "total_annual_cost_hs": "float32",
    # "annual_energy_costs_dhw": "float32",
    # "total_annual_cost_dhw": "float32",
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

SFH_MFH = {
    1: "SFH",
    2: "SFH",
    5: "MFH",
    6: "MFH"
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


def calculate_mean(grouped: pd.DataFrame, names: list, number_of_buildings: str):
    if grouped[number_of_buildings].sum() == 0:  # create nan row so it can be easily dropped later
        new_row = pd.Series(data=[np.nan] * len(grouped[names].columns), index=grouped[names].columns)
    else:
        weights = grouped[number_of_buildings] / grouped[number_of_buildings].sum()
        new_row = (grouped[names].T * weights).T.sum()
    return new_row


def create_representative_building(group: pd.DataFrame,
                                   column_name_with_numbers: str,
                                   merging_names: list,
                                   adding_names: list) -> pd.DataFrame:
    new_row = pd.DataFrame(columns=group.columns, index=[0])
    # representative air source HP building
    new_row.loc[0, merging_names] = calculate_mean(group,
                                                   names=merging_names,
                                                   number_of_buildings=column_name_with_numbers)
    new_row.loc[0, adding_names] = group.loc[:, adding_names].sum()

    # new name is first 5 letters of first name + heating system
    new_row.loc[0, "heating_medium"] = group.loc[:, "heating_medium"].values[-1]
    new_row.loc[:, "construction_period_start"] = group.loc[:, "construction_period_start"].values[-1]
    new_row.loc[:, "construction_period_end"] = group.loc[:, "construction_period_end"].values[-1]
    new_row.loc[0, "name"] = f"{str(group.loc[:, 'name'].iloc[0])}"
    new_row.loc[0, "index"] = group.loc[:, "index"].iloc[0]  # new index is the first index of merged rows
    return new_row


def filter_only_certain_region_buildings(df: pd.DataFrame, invert_city_name: str):
    mask = df["name"].astype(str).isin([name for name in df["name"].astype(str) if invert_city_name in name])
    return df.loc[mask, :]


def get_number_heating_systems(df: pd.DataFrame) -> dict:
    numbers = df.groupby("heating_system")["number_of_buildings"].sum()
    return numbers.to_dict()


def get_number_of_buildings_from_invert(invert_city_filter_name: str, country: str, year: int, scen: str) -> pd.DataFrame:
    hdf5_f = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data") / f"001_buildings_{country.lower()}_{scen}.hdf5"
    bc_df = hdf5_to_pandas(hdf5_f, f"BC_{year}", BUILDING_CLASS_COLUMNS)
    bssh_df = hdf5_to_pandas(hdf5_f, f"BSSH_{year}", BUILDING_SEGMENT_COLUMNS)
    # remove multiindex
    bssh_df.columns = bssh_df.columns.map(''.join)
    bc_df.columns = bc_df.columns.map(''.join)
    # change columns to series:
    bssh = bssh_df.apply(to_series)
    bc = bc_df.apply(to_series)
    bc["building_categories_index"] = bc["building_categories_index"].astype(int)
    # use only buildings from a specific city that represents a region in invert!
    if country.lower() == "spain":
        bc = filter_only_certain_region_buildings(bc, invert_city_filter_name)
        bssh = filter_only_certain_region_buildings(bssh, invert_city_filter_name)

    # remove duplicated rows of the bssh dataframe (don't know why they exist)
    duplicate_rows = bssh.drop(columns=["index", "name"]).duplicated()
    bssh_clean = bssh[~duplicate_rows].reset_index(drop=True)
    # replace heating system indices that are the same as the additional information like installation year is not of our interest:
    bssh_clean.loc[:, "heating_system"] = bssh_clean.loc[:, "heat_supply_system_index"].map(HEATING_SYSTEM_INDEX)

    grouped = bssh_clean.groupby("building_classes_index")
    x_df = bc.copy()
    for index, group in tqdm.tqdm(grouped, desc=f"adding heating systems to building categories in {year}"):
        # add total number of buildings:
        total_number_buildings = group["number_of_buildings"].sum()
        x_df.loc[x_df.loc[:, "index"] == index, "number_of_buildings"] = total_number_buildings
        # add the number of buildings with other energy carriers
        # numbers = get_number_heating_systems(df=group)
        # for carrier, number in numbers.items():
        #     x_df.loc[x_df.loc[:, "index"] == index, f"number_buildings_{carrier.replace(' ', '_')}"] = number

    # nan to zero and only consider residential buildings:
    final_bc = x_df.fillna(0).loc[x_df.loc[:, "building_categories_index"].isin(list(SFH_MFH.keys())), :].reset_index(
        drop=True)
    final_bc['name'] = final_bc['name'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return final_bc


def get_dynamic_calc_data(year: int, country: str, scen: str) -> pd.DataFrame:
    path_to_dynamic_data = Path(
        r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data") / f"001__dynamic_calc_data_bc_{year}_{country}_{scen}.npz"
    npz = np.load(path_to_dynamic_data)
    column_names = npz["arr_1"]
    data = npz["arr_0"]
    df = pd.DataFrame(data=data)
    df.columns = column_names.squeeze()
    return df


def get_probabilities_for_building_to_change(old_year: int, new_year: int, scen: str) -> (pd.DataFrame, pd.DataFrame):
    buildings_2020 = get_number_of_buildings_from_invert("Sevilla", "spain", old_year, scen=scen)
    buildings_2030 = get_number_of_buildings_from_invert("Sevilla", "spain", new_year, scen=scen)

    # for every building in 2020 the total number of buildings will be compared with the same bc index of 2030
    # the difference in number of buildings is and indicator of how probable it is that the building was renovated
    # or newly built.
    bc_new_pool = buildings_2030.loc[~buildings_2030["index"].isin(buildings_2020["index"].to_list()), :].copy()
    # based on the numbers of buildings it is decided if a building stays the same or a new building is drawn from the
    # 2030 building pool (bc_2030_new_pool).
    differenz = buildings_2020["number_of_buildings"] - buildings_2030.loc[
        buildings_2030["index"].isin(buildings_2020["index"].to_list()), "number_of_buildings"]
    # some of the buildings in 2020 will increase in numbers as they are renovated versions, thus the difference will
    # be negative. This building type should be included into the 2030 pool again with the difference as the number
    # of total buildings. This way other buildings from 2020 can become this building category.
    add_to_pool = buildings_2020.loc[differenz[differenz < 0].index, :]
    # adapt the number of buildings for the add to pool buildings from 2020:
    add_to_pool["number_of_buildings"] = differenz[differenz < 0] * -1
    bc_new_pool = pd.concat([bc_new_pool, add_to_pool], axis=0)
    bc_new_pool.loc[:, "construction_period"] = bc_new_pool.loc[:, "construction_period_start"].astype(str) + \
                                                "-" + bc_new_pool.loc[:, "construction_period_end"].astype(
        str)
    # now set the probabilities of the buildings with negative difference to 0
    differenz[differenz < 0] = 0
    # calc probabilities
    probs = differenz / buildings_2020["number_of_buildings"]
    probabilities = pd.concat([buildings_2020["index"], probs], axis=1).rename(
        columns={"number_of_buildings": "probability"})
    return probabilities, bc_new_pool





if __name__ == "__main__":
    # city names for spain are Sevilla (closest to Murcia) and for the Netherlands we don't filter because
    # there are no representative regions except amsterdam
    country = "Spain"
    choices, bc_2030_new_pool = get_probabilities_for_building_to_change(old_year=2020, new_year=2030)
    # with the choices go into the 2020 murcia df and for all building types that have choice=True select a new
    # building from the new pool:

    # load the old non clustered buildings:
    old_buildings = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") /
                                  "2020_combined_building_df_Murcia_non_clustered.xlsx")
    old_5R1C = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") /
                             "OperationScenario_Component_Building_Murcia_non_clustered_2020.xlsx")

    update_city_buildings(choices=choices,
                          new_building_pool=bc_2030_new_pool,
                          old_building_df=old_buildings,
                          old_5R1C_df=old_5R1C,
                          new_year=2030,
                          country=country)
