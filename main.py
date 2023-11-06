import numpy as np
import numpy.random
import osmapi
import osmnx as ox
import pandas as pd
from shapely.geometry import shape
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import h5py
from load_invert_data import get_number_of_buildings_from_invert_spain
from mosis_wonder import calc_premeter
from convert_to_5R1C import Create5R1CParameters
import warnings

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
    murcia_df = gpd.read_file(Path(r"input_data\30030.gpkg"))
    df = murcia_df.cx[region["west"]: region["east"], region["south"]: region["north"]].copy()
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
    check = False
    for period in construction_periods:
        if int(period.split("-")[0]) <= cell <= int(period.split("-")[1]):
            check = True
            # print(f"{cell} is between {period.split('-')[0]} and {period.split('-')[1]}")
            break
    if not check:
        period = "2002-2008"
        print(f"{cell} is in no period! {period} is chosen instead")
    return period


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


def add_invert_data_to_gdf_table(gdf: gpd.GeoDataFrame):
    # load invert table for Sevilla buildings
    df_invert = get_number_of_buildings_from_invert_spain()

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
    types = gdf["uso_princi"].unique()
    print(types)
    type_groups = gdf.groupby("uso_princi")
    complete_df = pd.DataFrame()
    for type, group in type_groups:
        # add construction year to the buildings that don't have one
        # group["ano_construccion"] = replace_nan_with_distribution(group, "ano_construccion")
        group["ano_constr"] = group["ano_constr"].astype(int)
        # add the number of stories of the building if not known
        # group["altura_maxima"] = replace_nan_with_distribution(group, "altura_maxima")

        # select a building type from invert based on the construction year and type
        group["invert_construction_period"] = group["ano_constr"].apply(
            lambda x: find_construction_period(construction_periods, x)
        )
        # invert_selection = filter_invert_data_after_type(type=type, invert_df=df_invert)  # does not work for residential (i have a mistake)

        # now select a representative building from invert for each building from Urban3r:
        for i, row in group.iterrows():
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
            complete_df = pd.concat([
                complete_df,
                pd.DataFrame(pd.concat([selected_building, row.drop(columns=["number_of_buildings"])], axis=0)).T
            ], axis=0
            )

    final_df = complete_df.reset_index(drop=True)
    return final_df


def get_parameters_from_dynamic_calc_data(df: pd.DataFrame) -> pd.DataFrame:
    # load dynamic calc data
    dynamic_calc = pd.read_csv(Path(r"input_data\dynamic_calc_data_bc_2020_Spain.csv"), sep=";")
    # map the CM_Factor and the Am_factor to the df through the bc_index:
    df = df.rename(columns={"index": "bc_index"})
    merged_df = df.merge(
        dynamic_calc.loc[:, ["bc_index", "CM_factor", "Am_factor"]],
        on="bc_index",
        how="inner"
    )
    return merged_df


def calculate_5R1C_necessary_parameters(df):
    # number of floors
    df.loc[:, "floors"] = df.loc[:, "altura_max"] + 1
    # height of the building
    df.loc[:, "height"] = (df.loc[:, "room_height"] + 0.3) * df.loc[:, "floors"]
    # adjacent area
    df.loc[:, "free wall area (m2)"] = df.loc[:, "free length (m)"] * df.loc[:, "height"]
    # not adjacent area
    df.loc[:, "adjacent area (m2)"] = (df.loc[:, "circumference (m)"] - df.loc[:, "free length (m)"]) * \
                                      df.loc[:, "height"]
    # total wall area
    df.loc[:, "wall area (m2)"] = df.loc[:, "circumference (m)"] * df.loc[:, "height"]
    # ration of adjacent to not adjacent (to compare it to invert later)
    df.loc[:, "percentage attached surface area"] = df.loc[:, "adjacent area (m2)"] / df.loc[:, "wall area (m2)"]
    df_return = get_parameters_from_dynamic_calc_data(df)

    # demographic information: number of persons per house is number of dwellings (numero_viv) from Urban3R times number
    # of persons per dwelling from invert
    df_return.loc[:, "person_num"] = df_return.loc[:, "numero_viv"] * df_return.loc[:, "number_of_persons_per_dwelling"]
    # building type:
    df_return.loc[:, "type"] = ["SFH" if i == 1 else "MFH" for i in df_return.loc[:, "numero_viv"]]
    return df_return


def create_boiler_excel(df: pd.DataFrame,
                        city_name: str):
    translation_dict = {
        "electricity": "Electric",
        "heat pump air": "Air_HP",
        "heat pump ground": "Ground_HP",
        "no heating": "no heating",
        "coal": "solids",
        "wood": "solids",
        "oil": "liquids",
        "gas": "gases",
        "district heating": "district heating"
    }
    boiler = pd.DataFrame(data=np.arange(1, df.shape[0] + 1), columns=["ID_Boiler"])
    boiler.loc[:, "type"] = [translation_dict[i] for i in df.loc[:, "heating_medium"]]
    boiler.to_excel(Path(r"output_data") / f"OperationScenario_Component_Boiler_{city_name}.xlsx")
    boiler.to_excel(
        Path(r"C:\Users\mascherbauer\PycharmProjects\FLEX\data\input_operation") / f"ECEMF_T4.3_{city_name}" /
        f"OperationScenario_Component_Boiler_{city_name}.xlsx"
    )


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


def create_behavior_excel(country: str):
    behavior_dict = {
        "ID_Behavior": 1,
        "id_people_at_home_profile": 1,
        "target_temperature_at_home_max": 27,
        "target_temperature_at_home_min": 20,
        "target_temperature_not_at_home_max": 27,
        "target_temperature_not_at_home_min": 20,
        "shading_solar_reduction_rate": 0.5,
        "shading_threshold_temperature": 30,
        "temperature_unit": "°C",
        "id_hot_water_demand_profile": 1,
        "hot_water_demand_annual": specific_DHW_per_person_EU(country),
        "hot_water_demand_unit": "Wh/person",
        "id_appliance_electricity_demand_profile": 1,
        "appliance_electricity_demand_annual": appliance_electricity_demand_per_person_EU(country),
        "appliance_electricity_demand_unit": "Wh/person",
        "id_vehicle_at_home_profile": 1,
        "id_vehicle_distance_profile": 1,
    }
    behavior = pd.DataFrame.from_dict(behavior_dict, orient="index").T
    behavior.to_excel(Path(r"output_data") / f"OperationScenario_Component_Behavior_{country}.xlsx")
    behavior.to_excel(
        Path(r"C:\Users\mascherbauer\PycharmProjects\FLEX\data\input_operation") / f"ECEMF_T4.3_{city_name}" /
                      f"OperationScenario_Component_Behavior_{country}.xlsx"
    )


def convert_to_float(column):
    return pd.to_numeric(column, errors="ignore")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    region = SUCINA
    city_name = region["city_name"]
    country_name = "Spain"
    shp_filename = Path(f"merged_osm_geom_{city_name}.shp")
    extended_shp_filename = Path(f"merged_osm_geom_extended_{city_name}.shp")
    # merge the dataframes and safe the shapefile to shp_filename:
    merge_osm_urban3r(output_filename=shp_filename, region=region)
    # add the adjacent length and circumference with mosis wonder:
    big_df = calc_premeter(input_lyr=shp_filename,
                           output_lyr=extended_shp_filename, )

    # combine the Urban3R information with the Invert database
    combined_df = add_invert_data_to_gdf_table(big_df)

    # turn them to numeric
    numeric_df = combined_df.apply(convert_to_float)
    # calculate all necessary parameters for the 5R1C model:
    final_df = calculate_5R1C_necessary_parameters(numeric_df)

    # create the dataframe with 5R1C parameters
    building_df, total_df = Create5R1CParameters(df=final_df).main()
    building_df.to_excel(
        Path(f"output_data") / f"OperationScenario_Component_Building_{city_name}_non_clustered.xlsx", index=False
    )
    print("saved OperationScenario_Component_Building to xlsx")
    total_df.loc[:, "ID_Building"] = np.arange(1, total_df.shape[0] + 1)
    # add representative point for each building
    total_df['rep_point'] = total_df['geometry'].apply(lambda x: x.representative_point())
    total_df.to_excel(
        Path(f"output_data") / f"combined_building_df_{city_name}_non_clustered.xlsx",
        index=False
    )
    total_df.to_excel(
        Path(r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects") / f"ECEMF_T4.3_{city_name}" /
        f"combined_building_df_{city_name}_non_clustered.xlsx",
        index=False
    )
    print("saved dataframe with all information to xlsx")

    # create csv file with coordinates and shp file with dots to check in QGIS
    coordinate_df = gpd.GeoDataFrame(total_df[["rep_point", "ID_Building"]]).set_geometry("rep_point")
    coordinate_df.to_file(Path(r"output_data") / f"building_coordinates_{city_name}.shp", driver="ESRI Shapefile")
    coordinate_df.to_csv(Path(r"output_data") / f"Building_coordinates_{city_name}.csv", index=False)
    coordinate_df.to_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects") / f"ECEMF_T4.3_{city_name}"
                         / f"Building_coordinates_{city_name}.csv", index=False)

    # create the boiler table for the 5R1C model:
    create_boiler_excel(df=final_df,
                        city_name=city_name)
    # create Behavior table for 5R1C model:
    create_behavior_excel(country=country_name)

