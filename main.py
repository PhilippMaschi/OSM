import shutil

import numpy as np
import numpy.random
import pandas as pd
import tqdm
from shapely.geometry import shape
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from load_invert_data import get_number_of_buildings_from_invert, get_probabilities_for_building_to_change, \
    update_city_buildings, calculate_5R1C_necessary_parameters
from cluster_buildings import main as cluster_main
from mosis_wonder import calc_premeter
from convert_to_5R1C import Create5R1CParameters
import warnings
import random


# Suppress the FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
# Coordinates are EPSG: 4326 !!! WGS84
BASE_EPSG = 4326
LEEUWARDEN = {"north": 53.2178674080337,
              "south": 53.1932515262881,
              "east": 5.82625369878255,
              "west": 5.76735091362368,
              "city_name": "Leeuwarden"}

KWIDZYN = {"north": 53.7487987337842,
           "south": 53.7164945915439,
           "east": 18.9535734503598,
           "west": 18.9095466059208,
           "city_name": "Kwidzyn"}

BAARD = {"north": 53.1620856552012,
         "south": 53.1314293502190,
         "east": 5.68494449734352,
         "west": 5.64167343762892,
         "city_name": "Baard"}

RUMIA = {"north": 54.5877326995821,
         "south": 54.5560197018211,
         "east": 18.4243116632352,
         "west": 18.3784147154927,
         "city_name": "Rumia"}

SUCINA = {"north": 37.9007686931277,
          "south": 37.8459872260350,
          "east": -0.929828651725462,
          "west": -0.976709171036036,
          "city_name": "Sucina"}

MURCIA = {"north": 37.9988137604873,
          "south": 37.9656536677982,
          "east": -1.10710096876283,
          "west": -1.13912542769010,
          "city_name": "Murcia"}

TRANSLATION_DICT = {
    "Almacenaje": "storage",
    "Aparcamiento": "parking",
    "Residencial": "residential",
    "Público": "public",
    "Piscinas": "swimming pool",
    "EnseñanzaCultural": "cultural education",
    "HoteleroRestauración": "hotel",
    "IndustrialResto": "industrial site",
    "OtrosNoCal": "other no cal",
    "OtrosCal": "other cal",
    "Industrial": "industrial",
    "Comercial": "commercial",
    "VincViv": "social housing",
    "Deportivo": "sport",
    "Común": "shared",
    "Oficinas": "offices",
}

BUILDING_FILTER = [
    # "hotel",
    # "other no cal",
    # "other cal",
    # "commercial",
    # "social housing",
    # "shared",
    # "offices",
    "residential",
    # "public",
    # "cultural education"
]

USO_PRINCIPAL_TO_INVERT_TYPE = {
    # "hotel": "Hotel & restaurants",
    # "commercial": "Wholesale",
    "social housing": "MFH",
    "shared": "",
    "offices": "Private offices",
    "residential": ["SFH", "MFH"],
    "public": "Public",
    "cultural education": "Education"
}


def get_osm_gdf(city: dict):
    gdf = gpd.read_file(Path("input_data/OpenStreetMaps") / f"{city['city_name']}.gpkg")
    return gdf


def get_osm_building_numbers(city: dict, big_df: pd.DataFrame):
    # extract the buildings within the bounding box
    buildings = get_osm_gdf(city)
    new_df = pd.DataFrame(buildings['building']).reset_index(drop=True)
    new_df["city"] = city["city_name"]
    df = pd.concat([big_df, new_df], axis=0)
    # extract the building types from the buildings
    print(f"totoal number of buildings in {city['city_name']}: {len(buildings)} \n")
    return df


def load_invert_spain_data():
    df = pd.read_csv(Path(
        r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data\Building_Classes_2020_Spain.csv", sep=";"
    ))
    return df


def get_urban3r_murcia_gdf(region: dict):
    murcia_df = gpd.read_file(Path(r"input_data\Urban3R") / f"{region['city_name']}.gpkg")
    df = murcia_df.copy()
    df["number_of_buildings"] = 1
    print(f"URBAN3R buildings in database total: {df['number_of_buildings'].sum()}")
    df["uso_principal"] = df["uso_principal"].replace(TRANSLATION_DICT)
    filtered_gdf = df[df["uso_principal"].isin(BUILDING_FILTER)]
    print(f"URBAN3R buildings in database after filtering: {filtered_gdf['number_of_buildings'].sum()}")
    gdf = filtered_gdf.to_crs("epsg:3035")
    return gdf


def load_urban3r_murcia(region: dict) -> gpd.GeoDataFrame:
    gdf = get_urban3r_murcia_gdf(region)
    # estimate the building height, 0 means only one floor --> 3m height?
    gdf["building_height"] = (gdf["altura_maxima"] + 1) * 3
    # get building footprint
    gdf["building_footprint"] = gdf.area
    # volume of the buildings: area * height
    gdf["building_volume"] = gdf["building_footprint"] * gdf["building_height"]

    # habe nicht die richtigen epsg um die koordinaten zu transformieren, darum nehme ich das längen und Breiten
    # Verhältnis
    gdf['length'] = gdf.length
    gdf['width'] = gdf['building_footprint'] / gdf['length']
    gdf["surface area"] = 2 * gdf['length'] * gdf["building_height"] + \
                          2 * gdf['width'] * gdf["building_height"] + \
                          gdf["building_footprint"]  # as roof area aprox.
    gdf["volume surface ratio"] = gdf["building_volume"] / gdf["surface area"]
    return gdf


def merge_osm_urban3r(output_filename: Path, region: dict) -> None:
    print("merging OSM data with urban3r")
    urban3r = get_urban3r_murcia_gdf(region)
    urban3r["id_urban3r"] = np.arange(urban3r.shape[0])
    urban3r.to_file(f"urban3r_{region['city_name']}.shp", driver="ESRI Shapefile")
    osm = get_osm_gdf(region)
    osm_gdf = osm.to_crs("epsg:3035").copy()
    osm_gdf["id_osm"] = np.arange(osm_gdf.shape[0])

    # take the representative points from OSM and check how many of them are inside the shapes of urban3r
    osm_help = osm_gdf.copy()
    osm_help["help_geometry"] = osm_help["geometry"]
    osm_help["geometry"] = osm_help.representative_point()
    merged = gpd.sjoin(urban3r, osm_help, how='inner', op='contains')

    # take the same dataset and use the osm geometry:
    if "ways" in merged.columns:
        merged_osm_geom = merged.drop(columns=["geometry", "ways"]).rename(columns={"help_geometry": "geometry"})
    else:
        merged_osm_geom = merged.drop(columns=["geometry"]).rename(columns={"help_geometry": "geometry"})
    merged_osm_geom = merged_osm_geom[merged_osm_geom.geometry.type.isin(["Polygon"])]
    merged_osm_geom.to_file(output_filename, driver="ESRI Shapefile")
    print(f"OSM number of buildings total: {osm_gdf.shape[0]}")
    print(f"OSM building number after merging: {merged_osm_geom.shape[0]}")
    return None


def replace_nan_with_distribution(gdf: pd.DataFrame, column_name: str) -> gpd.GeoDataFrame:
    """
    for buildings where there are nan in a column we take the distribution from the
    buildings where we know it:
    """
    help_series = gdf.loc[:, column_name]
    distribution = help_series.dropna().value_counts(normalize=True)
    missing = help_series.isnull()
    gdf.loc[missing, column_name] = np.random.choice(distribution.index,
                                                     size=len(missing),
                                                     p=distribution.values)
    columns_with_different_lengths = [col for col in gdf.columns if len(gdf[col]) != len(gdf[gdf.columns[0]])]
    return gdf.reset_index(drop=True)


def find_construction_period(construction_periods, cell) -> str:
    parsed_intervals = []
    for interval in construction_periods:
        start, end = map(int, interval.split('-'))
        parsed_intervals.append((start, end))

    for start, end in parsed_intervals:
        if start <= cell <= end:
            return f"{start}-{end}"

    # If the year doesn't fall within any interval, find the closest interval
    closest_interval = None
    smallest_difference = float('inf')
    for start, end in parsed_intervals:
        # Check the distance to the start and end of each interval
        start_diff = abs(start - cell)
        end_diff = abs(end - cell)

        # Update the closest interval if a smaller difference is found
        if start_diff < smallest_difference:
            closest_interval = (start, end)
            smallest_difference = start_diff
        if end_diff < smallest_difference:
            closest_interval = (start, end)
            smallest_difference = end_diff
    return f"{closest_interval[0]}-{closest_interval[1]}"



def filter_invert_data_after_type(type: str, invert_df: pd.DataFrame) -> pd.DataFrame:
    # filter the invert data based on the uso_principal
    if type == "hotel":
        # filter invert data after Sevilla:
        mask = invert_df["name"].isin([name for name in invert_df["name"] if "Hotel & restaurants" in name])
        selection = invert_df.loc[mask, :]
    elif type == "commercial":
        mask = invert_df["name"].isin([name for name in invert_df["name"] if "Wholesale" in name or "Other" in name])
        selection = invert_df.loc[mask, :]
    elif type == "social housing":
        mask = invert_df["name"].isin([name for name in invert_df["name"] if "MFH" in name])
        selection = invert_df.loc[mask, :]
    elif type == "shared":
        mask = invert_df["name"].isin([name for name in invert_df["name"] if "MFH" in name or "SFH" in name])
        selection = invert_df.loc[mask, :]
    elif type == "offices":
        mask = invert_df["name"].isin([name for name in invert_df["name"] if "Private offices" in name])
        selection = invert_df.loc[mask, :]
    elif type == "residential":
        mask = invert_df["name"].isin([name for name in invert_df["name"] if "MFH" in name or "SFH" in name])
        selection = invert_df.loc[mask, :]
    elif type == "public":
        mask = invert_df["name"].isin([name for name in invert_df["name"] if "Public" in name])
        selection = invert_df.loc[mask, :]
    elif type == "cultural education":
        mask = invert_df["name"].isin([name for name in invert_df["name"] if "Education" in name])
        selection = invert_df.loc[mask, :]
    else:
        print(f"uso principal not defined in the code: {type}")
        return None
    return selection


def add_invert_data_to_gdf_table(gdf: gpd.GeoDataFrame, country: str, year: int, scen: str):
    # load invert table for Sevilla buildings
    df_invert = get_number_of_buildings_from_invert(country=country,
                                                    year=year,
                                                    scen=scen)
    df_invert.loc[:, "construction_period"] = df_invert.loc[:, "construction_period_start"].astype(str) + "-" + \
                                              df_invert.loc[:, "construction_period_end"].astype(str)
    # drop the rows where buildings are built between 1900-1980 etc.
    df_invert = df_invert.loc[df_invert.loc[:, "construction_period"] != "1900-1980", :]
    df_invert = df_invert.loc[df_invert.loc[:, "construction_period"] != "2007-2008", :]
    df_invert = df_invert.loc[df_invert.loc[:, "construction_period"] != "1981-2006", :]
    # group invert data after construction period:
    construction_periods = list(df_invert.loc[:, "construction_period"].unique())
    gdf = gdf.dropna(axis=1)
    # calculate the ground area
    gdf["area"] = gdf.area
    # for each building in the gdf extract the type (SFH or MFH or something else) and the year of construction

    # add construction year to the buildings that don't have one
    # group["ano_construccion"] = replace_nan_with_distribution(group, "ano_construccion")
    gdf["ano_constr"] = gdf["ano_constr"].astype(int)

    # select a building type from invert based on the construction year and type
    gdf["invert_construction_period"] = gdf["ano_constr"].apply(
        lambda x: find_construction_period(construction_periods, x)
    )

    df_list = []
    np.random.seed(42)
    # now select a representative building from invert for each building from Urban3r:
    for i, row in tqdm.tqdm(gdf.iterrows(), desc="selecting representative buildings from invert"):
        close_selection = df_invert.loc[
                          df_invert["construction_period"] == row['invert_construction_period'], :
                          ]
        # distinguish between SFH and MFH
        if row["tipologia_"] == "Unifamiliar":
            sfh_or_mfh = "SFH"
        else:
            sfh_or_mfh = "MFH"
        mask = close_selection["name"].isin([name for name in close_selection["name"] if sfh_or_mfh in name])
        type_selection = close_selection.loc[mask, :]
        if type_selection.shape[0] == 0:  # if only SFH or MFH available from invert take this data instead
            type_selection = close_selection
        # now get select the building after the probability based on the distribution of the number of buildings
        # in invert:
        distribution = type_selection["number_of_buildings"].astype(float) / \
                       type_selection["number_of_buildings"].astype(float).sum()
        # draw one sample from the distribution
        random_draw = np.random.choice(distribution.index, size=1, p=distribution.values)[0]
        selected_building = type_selection.loc[random_draw, :]
        # delete number of buildings because this number is 1 for a single building
        df_list.append(pd.DataFrame(pd.concat([selected_building, row.drop(columns=["number_of_buildings"])], axis=0)).T)

    final_df = pd.concat(df_list, axis=0).reset_index(drop=True)

    return final_df


def create_boiler_excel() -> pd.DataFrame:
    boiler_dict = {
        "ID_Boiler": [1, 2, 3, 4, 5],
        "type": ["Electric", "Air_HP", "Ground_HP", "no heating", "gas"],
        "carnot_efficiency_factor": [1, 0.4, 0.45, 1, 0.95]
    }
    boiler_df = pd.DataFrame(boiler_dict)
    return boiler_df



def load_european_population_df(country: str) -> int:
    """
    :param country: name of the country
    :return: number of people living in the country
    """
    population = pd.read_excel(
        Path(r"input_data\Europe_population_2020.xlsx"),
        sheet_name="Sheet 1",
        engine="openpyxl",
        skiprows=8,
    ).drop(
        columns=["Unnamed: 2"]
    )
    population.columns = ["country", "population"]
    try:
        return population.loc[population.loc[:, "country"] == country, "population"].values[0]
    except:
        print(f"country is not found. Choose from following list: \n "
              f"{population['country'].unique()}")


def load_european_consumption_df(country: str) -> pd.DataFrame:
    """

    :param country:
    :return:
    """
    consumption = pd.read_excel(
        Path(r"input_data\Europe_residential_energy_consumption_2020.xlsx"),
        sheet_name="Sheet 1",
        engine="openpyxl",
        skiprows=9,
    )
    consumption.columns = ["country", "type", "consumption (TJ)"]
    consumption["type"] = consumption["type"].str.replace(
        "Final consumption - other sectors - households - energy use - ", ""
    ).replace(
        "Final consumption - other sectors - households - energy use", "total"
    )
    consumption_df = consumption.copy()
    # replace : with 0
    consumption_df["consumption (TJ)"] = consumption_df["consumption (TJ)"].apply(
        lambda x: float(str(x).replace(":", "0")))
    # convert terra joule in kW
    consumption_df["consumption (GWh)"] = consumption_df["consumption (TJ)"].astype(float) / 3_600 * 1_000

    # drop tJ column
    consumption_df = consumption_df.drop(columns=["consumption (TJ)"])
    try:
        return consumption_df.loc[consumption_df.loc[:, "country"] == country, :]
    except:
        print(f"country is not found. Choose from following list: \n "
              f"{consumption_df['country'].unique()}")


def specific_DHW_per_person_EU(country: str) -> float:
    """
    :param country: country name
    :return: returns the consumption for DHW per person in Wh
    """
    consumption = load_european_consumption_df(country).query("type == 'water heating'")["consumption (GWh)"].values[0]
    population = load_european_population_df(country)
    return consumption / population * 1_000 * 1_000 * 1_000


def appliance_electricity_demand_per_person_EU(country: str) -> float:
    consumption = load_european_consumption_df(country).query("type == 'lighting and electrical appliances'")["consumption (GWh)"].values[0]
    population = load_european_population_df(country)
    return consumption / population * 1_000 * 1_000 * 1_000


def generate_heating_schedule():
    # Constants
    start_afternoon = 17  # 5 PM
    end_afternoon = 22  # 10 PM
    start_morning = 6  # 6 AM
    end_morning = 9  # 9 AM

    # Initialize all hours to 0 (heating off)
    schedule = [0] * 8760

    for day in range(365):
        start_index = day * 24
        # Determine morning heating
        if random.random() < 0.5:  # 50% chance
            morning_heating_hour = random.randint(start_morning, end_morning)
            morning_heating_index = start_index + morning_heating_hour
            schedule[morning_heating_index] = 1

        # Determine afternoon heating
        heating_duration = random.randint(1, 3)
        afternoon_heating_start = random.randint(start_afternoon, end_afternoon - heating_duration)

        for i in range(heating_duration):
            afternoon_heating_index = start_index + afternoon_heating_start + i
            schedule[afternoon_heating_index] = 1

    return schedule


def create_people_at_home_profiles(city_name: str) -> pd.DataFrame:
    id_hour = np.arange(1, 8761)
    people_at_home_profile_1 = np.full(shape=(8760,), fill_value=1)
    # the second profile is used for people who have no heating
    people_at_home_profile_2 = np.full(shape=(8760,), fill_value=1)  # target temp is 10°C so this doesnt matter

    # all other profiles are used for people who have direct electric heating and use it only for ~3 hours when
    # they come home and possibly for 1 hour in the morning. The heating will be turned on between 5 and 10 in the night
    # randomly chosen for a random time (1-3) hours. In the morning the heating will be turned on randomly between,
    # 6 and 9 o'clock with a 30% probability every day.
    # create 50 different profiles:

    # create excel OperationScenario_BehaviorProfile:
    df = pd.DataFrame(data=id_hour, columns=["id_hour"])
    df["people_at_home_profile_1"] = people_at_home_profile_1
    df["people_at_home_profile_2"] = people_at_home_profile_2

    for i in range(50):
        df[f"people_at_home_profile_{i+3}"] = generate_heating_schedule()

    hot_water_profile = pd.read_excel(
        Path(r"input_data") / f"FLEX_scenario_files_{city_name}" / "HotWaterProfile.xlsx"
    )
    df["hot_water_demand_profile_1"] = hot_water_profile
    appliance_profile = pd.read_excel(
        Path(r"input_data") / f"FLEX_scenario_files_{city_name}" / "Appliance_Profile.xlsx"
    )
    df["appliance_electricity_demand_profile_1"] = appliance_profile
    df["vehicle_hat_home_profile_1"] = np.zeros(shape=(8760,))
    df["vehicle_distance_profile_1"] = np.zeros(shape=(8760,))

    return df


def create_behavior_excel(country: str):
    behavior_dict = {
        "ID_Behavior": [1, 2, 3],
        "id_people_at_home_profile_min": [1, 2, 3],
        "id_people_at_home_profile_max": [1, 2, 53],
        "target_temperature_at_home_max": [27, 27, 27],
        "target_temperature_at_home_min": [20, 0, 18],
        "target_temperature_not_at_home_max": [27, 27, 27],
        "target_temperature_not_at_home_min": [20, 0, 10],
        "shading_solar_reduction_rate": [0.5, 0.5, 0.5],
        "shading_threshold_temperature": [30, 30, 30],
        "temperature_unit": ["°C", "°C", "°C"],
        "id_hot_water_demand_profile": [1, 1, 1],
        "hot_water_demand_annual": [specific_DHW_per_person_EU(country), specific_DHW_per_person_EU(country), specific_DHW_per_person_EU(country)],
        "hot_water_demand_unit": ["Wh/person", "Wh/person", "Wh/person"],
        "id_appliance_electricity_demand_profile": [1, 1, 1],
        "appliance_electricity_demand_annual": [appliance_electricity_demand_per_person_EU(country), appliance_electricity_demand_per_person_EU(country), appliance_electricity_demand_per_person_EU(country)],
        "appliance_electricity_demand_unit": ["Wh/person", "Wh/person", "Wh/person"],
        "id_vehicle_at_home_profile": [1, 1, 1],
        "id_vehicle_distance_profile": [1, 1, 1],
    }
    behavior = pd.DataFrame.from_dict(behavior_dict, orient="index").T
    return behavior


def convert_to_float(column):
    return pd.to_numeric(column, errors="ignore")


def get_related_5R1C_parameters(df: pd.DataFrame, year: int, country_name: str, scen: str) -> (pd.DataFrame, pd.DataFrame):
    # calculate all necessary parameters for the 5R1C model:
    final_df = calculate_5R1C_necessary_parameters(df, year, country_name, scen)

    # create the dataframe with 5R1C parameters
    building_df, total_df = Create5R1CParameters(df=final_df).main()
    return building_df, total_df


def create_2020_baseline_building_distribution(region: dict,
                                               city_name: str,
                                               country_name: str,
                                               scen: str
                                               ):
    year = 2020
    shp_filename = Path(f"merged_osm_geom_{city_name}.shp")
    extended_shp_filename = Path(f"merged_osm_geom_extended_{city_name}.shp")
    # merge the dataframes and safe the shapefile to shp_filename:
    if not extended_shp_filename.exists():
        merge_osm_urban3r(output_filename=shp_filename, region=region)
        # add the adjacent length and circumference with mosis wonder:
        big_df = calc_premeter(input_lyr=shp_filename,
                               output_lyr=extended_shp_filename)
    else:
        big_df = gpd.read_file(extended_shp_filename)

    # combine the Urban3R information with the Invert database
    combined_df = add_invert_data_to_gdf_table(big_df,
                                               country=country_name,
                                               
                                               year=year,
                                               scen=scen)

    # turn them to numeric
    numeric_df = combined_df.apply(convert_to_float)

    building_df, total_df = get_related_5R1C_parameters(df=numeric_df, year=year, country_name=country_name, scen=scen)

    # save the building df and total df for 2020 once. These dataframes will be reused for the following years:
    building_df.to_excel(
        Path(f"output_data") / f"OperationScenario_Component_Building_{city_name}_non_clustered_{year}_{scen}.xlsx",
        index=False
    )
    print("saved OperationScenario_Component_Building to xlsx")
    total_df.loc[:, "ID_Building"] = np.arange(1, total_df.shape[0] + 1)
    # add representative point for each building
    total_df['rep_point'] = total_df['geometry'].apply(lambda x: x.representative_point())
    total_df.to_excel(
        Path(f"output_data") / f"{year}_{scen}_combined_building_df_{city_name}_non_clustered.xlsx",
        index=False
    )
    print("saved dataframe with all information to xlsx")

    # create csv file with coordinates and shp file with dots to check in QGIS
    coordinate_df = gpd.GeoDataFrame(total_df[["rep_point", "ID_Building"]]).set_geometry("rep_point")
    coordinate_df.to_file(Path(r"output_data") / f"{scen}_building_coordinates_{city_name}.shp", driver="ESRI Shapefile")
    coordinate_df.to_csv(Path(r"output_data") / f"{scen}_Building_coordinates_{city_name}.csv", index=False)


def save_to_all_years_in_flex_folders(df_to_save: pd.DataFrame, years: list, filename: str, scenarios: list, city: str):
    for s in scenarios:
        for y in years:
            folder = Path("output_data") / f"ECEMF_T4.3_{city}_{y}_{s}"
            df_to_save.to_excel(folder / f"{filename}", index=False)


def copy_flex_input_files_to_year_runs(orig_file_location: Path, destination_path: Path):
    files_to_copy = [
        "OperationScenario_Component_HeatingElement.xlsx",
        "OperationScenario_Component_PV.xlsx",
        "OperationScenario_RegionWeather.xlsx",
        "OperationScenario_DrivingProfile_Distance.csv",
        "OperationScenario_DrivingProfile_ParkingHome.csv",
        "OperationScenario_Component_Region.xlsx",
        "OperationScenario_Component_SpaceCoolingTechnology.xlsx",
        "OperationScenario_EnergyPrice.xlsx",
        "OperationScenario_Component_Battery.xlsx",
        "OperationScenario_Component_HotWaterTank.xlsx",
        "OperationScenario_Component_SpaceHeatingTank.xlsx",
        "OperationScenario_Component_EnergyPrice.xlsx",
        "OperationScenario_Component_Vehicle.xlsx"
    ]
    for file in files_to_copy:
        shutil.copy(src=orig_file_location / file, dst=destination_path / file)


def create_flex_input_folders(city: str, years: list, scenarios: list):
    flex_scenario_folder = Path(r"input_data") / f"FLEX_scenario_files_{city}"
    for y in years:
        for s in scenarios:
            folder = Path("output_data") / f"ECEMF_T4.3_{city}_{y}_{s}"
            folder.mkdir(exist_ok=True)
            copy_flex_input_files_to_year_runs(
                orig_file_location=flex_scenario_folder,
                destination_path=folder,
            )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    region = MURCIA
    city_name = region["city_name"]
    country_name = "Spain"
    years = [2030, 2040, 2050]
    scenarios = ["H", "M"] #"moderate_eff",
    for scenario in scenarios:
        # generate the baseline:
        create_2020_baseline_building_distribution(region=region,
                                                   city_name=city_name,
                                                   country_name=country_name,
                                                   scen=scenario)
        # now generate iteratively the building files for the years based on the baseline:
        np.random.seed(42)
        for i, new_year in enumerate(years):
            if i == 0:
                old_year = 2020
            else:
                old_year = years[i-1]
            propb, bc_new_pool = get_probabilities_for_building_to_change(old_year=old_year,
                                                                          new_year=new_year,
                                                                          scen=scenario,
                                                                          country=country_name,
                                                                          city=city_name)
            # with the choices go into the 2020 murcia df and for all building types that have choice=True select a new
            # building from the new pool:

            # load the old non clustered buildings:
            old_buildings = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") /
                                          f"{old_year}_{scenario}_combined_building_df_Murcia_non_clustered.xlsx")
            old_5R1C = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") /
                                     f"OperationScenario_Component_Building_Murcia_non_clustered_{old_year}_{scenario}.xlsx")

            update_city_buildings(probability=propb,
                                  new_building_pool=bc_new_pool,
                                  old_building_df=old_buildings,
                                  old_5R1C_df=old_5R1C,
                                  new_year=new_year,
                                  country=country_name,
                                  scen=scenario,
                                  city=city_name)

    # prepare the FLEX runs:
    create_flex_input_folders(city=region["city_name"], years=[2020] + years, scenarios=scenarios)
    # create the boiler table for the 5R1C model:
    boiler_df = create_boiler_excel()
    # create Behavior table for 5R1C model:
    behavior_df = create_behavior_excel(country=country_name)
    # create the stay at home profiles as the people with direct electric heating will only use it rarely which is
    # reflected in the target temperatures of ID Behavior 2 in the behavior table
    behavior_profile = create_people_at_home_profiles(city_name=city_name)
    # create FLEX starting folders:
    save_to_all_years_in_flex_folders(df_to_save=boiler_df,
                                      years=[2020] + years,
                                      filename="OperationScenario_Component_Boiler.xlsx",
                                      scenarios=scenarios,
                                      city=city_name)
    save_to_all_years_in_flex_folders(df_to_save=behavior_df,
                                      years=[2020] + years,
                                      filename="OperationScenario_Component_Behavior.xlsx",
                                      scenarios=scenarios,
                                      city=city_name)
    save_to_all_years_in_flex_folders(df_to_save=behavior_profile,
                                      years=[2020] + years,
                                      filename="OperationScenario_BehaviorProfile.xlsx",
                                      scenarios=scenarios,
                                      city=city_name)


    #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # after all this cluster_buildings.py has to be run to get the start data for the ECEMF runs done in FLEX.
    #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("starting clustering procedure")
    cluster_main(region=city_name, years=[2020]+years, scenarios=scenarios)
    # Then the data has to be copied from the respective FLEX folder from OSM/output_data to the FLEX repo.

