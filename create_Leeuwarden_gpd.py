import geopandas as gpd
from pathlib import Path
import pandas as pd
from main import BASE_EPSG, LEEUWARDEN, find_construction_period, get_parameters_from_dynamic_calc_data
from tqdm import tqdm
from load_invert_data import get_number_of_buildings_from_invert
from mosis_wonder import calc_premeter
from convert_to_5R1C import Create5R1CParameters
import numpy as np

columns_2_drop = [
    'b3_bag_bag_overlap',
    'b3_mutatie_ahn3_ahn4',
    'b3_nodata_fractie_ahn3',
    'b3_nodata_radius_ahn3',
    'b3_nodata_radius_ahn4',
    'b3_nodata_fractie_ahn4',
    'b3_puntdichtheid_ahn3',
    'b3_puntdichtheid_ahn4',
    'b3_pw_bron',
    'b3_pw_datum',
    'b3_pw_selectie_reden',
    'b3_rmse_lod12',
    'b3_rmse_lod13',
    'b3_rmse_lod22',
    'b3_val3dity_lod12',
    'b3_val3dity_lod13',
    'b3_val3dity_lod22',
    'b3_volume_lod12',
    'b3_volume_lod13',
    'begingeldigheid',
    'documentdatum',
    'documentnummer',
    'eindgeldigheid',
    'eindregistratie',
    'geconstateerd',
    'identificatie',
    'tijdstipeindregistratielv',
    'tijdstipinactief',
    'tijdstipinactieflv',
    'tijdstipnietbaglv',
    'tijdstipregistratie',
    'tijdstipregistratielv',
    'b3_dd_id',
    'b3_pand_deel_id',
    'b3_h_70p',
    'b3_kwaliteitsindicator',
]

filter_dict = {
    'status': ['Pand in gebruik', 'Verbouwing pand'],
}

translation_dict = {
    "b3_dak_type": "roof_type",
    'b3_kas_warenhuis': "warehouse",
    'b3_opp_buitenmuur': "outer_wall_area",
    'b3_opp_dak_plat': "flat_roof_area",
    'b3_opp_dak_schuin': "sloping_roof_area",
    'b3_opp_grond': "area",  # ground_area
    'b3_opp_scheidingsmuur': "partition_wall_area",
    'b3_reconstructie_onvolledig': "incomplete_reconstruction",
    'b3_volume_lod22': "building_volume",
    'oorspronkelijkbouwjaar': "original_year_of_construction",
    'voorkomenidentificatie': "occurrence_identification",
    'b3_h_maaiveld': "ground_level",
    "b3_h_dak_50p": "median_roof_height",
    "b3_h_dak_min": "minimum_roof_height",
    "b3_h_max": "maximum_roof_height",
    'b3_bouwlagen': "floors",

}

osm_building_types_filter = {
    'house': True,
    'yes': True,
    'apartments': True,
    'commercial': False,
    'school': False,
    'garage': False,
    'construction': False,
    'industrial': False,
    'shed': False,
    'retail': False,
    'office': False,
    'grandstand': False,
    'sports_centre': False,
    'service': False,
    'train_station': False,
    'public': False,
    'church': False,
    'almshouse': False,
    'roof': False,
    'gatehouse': False,
    'trade_pavilion': False,
    'warehouse': False,
    'bank': False,
    'military': False,
    'rectory': False,
    'hotel': False,
    'orphanage': False,
    'library': False,
    'civic': False,
    'garden_house': False,
    'synagogue': False,
    'mixed_use': False,
    'prison': False,
    'barracks': False,
    'fire_station': False,
    'boathouse': False,
    'hospital': False,
    'stable': False,
    'residential': True,
    'military_hospital': False,
    'barn': False,
    'windmill': False,
    'university': False,
    'college': False,
    'greenhouse': False
}


def merge_gpkg_files():
    # AFTER finding the database on QGIS and cutting it from there this function is not needed anymore! produes
    # gdf with missing data (building heights) were missing.
    """{
    "identificationInfo": {
    "citation": {
    "title": "3D BAG",
    "date": "2023-10-08",
    "dateType": "creation",
    "edition": "v2023.10.08",
    "identifier": "3652fa0c-6566-11ee-bc83-858db9e376fa"
},"""
    print("merging gpk files")
    path_2_files = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data\TU_Delft")
    files = list(path_2_files.glob("*.gpkg"))
    big_df = gpd.GeoDataFrame()
    for i, file in tqdm(enumerate(files)):
        if "lod2d" in file.stem.lower():
            continue
        df = gpd.read_file(file)
        if i == 0:
            crs = df.crs
        # clean and translate the dataframe and drop buildings that are not in use:
        dropped_df = df.drop(columns=columns_2_drop)
        translated_df = dropped_df.rename(columns=translation_dict)
        mask = pd.concat([df[col].isin(values) for col, values in filter_dict.items()], axis=1).all(axis=1)
        filtered_df = translated_df[mask]
        # drop all warehouses:
        filtered_df_2 = filtered_df[~filtered_df["warehouse"]]
        big_df = pd.concat([big_df, filtered_df_2], axis=0)

    new_df = big_df.to_crs(BASE_EPSG)
    leeuwarden_df = new_df.cx[LEEUWARDEN["west"]: LEEUWARDEN["east"], LEEUWARDEN["south"]: LEEUWARDEN["north"]]
    leeuwarden_df.to_file(Path(r"input_data/TU_Delft") / f"Leeuwarden.gpkg", driver="GPKG")
    print("saved Leeuwarden gpkg file")


def manual_removal_of_large_non_residential_buildings(dataframe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ids_to_remove = [
        23693,
        26054,
        12748,
        13376,
        6703,
        25790,
        4058,
        4237,
        32,
        23327,
        16160,
        16414,
        24463,
        18664,
        2905,
        2911,
        2936,
        2945,
        2897,
        2901,
        24403,
        2898,
        2904,
        2895,
        2915,
        2910,
        2937,
        2913,
        18624,
        17990,
        17364,
        18480,
        1140,
        2663,
        1819,
        3924,
        3897,
        3195,
        18581,
        17978,
        17822,
        18392,
        18406,
        17237,
        22263,

    ]
    # create set out of list in case buildigns were selected twice:
    to_remove = set(ids_to_remove)
    df_filtered = dataframe[~dataframe['OSM_ID'].isin(to_remove)]
    print("manually removed industrial buildings from dataset")
    return df_filtered


def add_osm_information_to_leeuwarden():
    # load OSM
    osm_df = gpd.read_file(Path(r"input_data/OpenStreetMaps") / "Leeuwarden.gpkg")
    osm_df["help_geometry"] = osm_df["geometry"]
    osm_df["geometry"] = osm_df.representative_point()
    osm_df.drop(columns=["help_geometry"]).to_file(Path(r"input_data") / "Leeuwarden_OSM.gpkg", driver="GPKG")
    # load TU Delft data
    lod12d = gpd.read_file(Path(r"input_data/TU_Delft") / "LoD12D.gpkg")
    if lod12d.crs.to_epsg() != BASE_EPSG:
        lod12d.to_crs(BASE_EPSG)
    leeuwarden = lod12d.cx[LEEUWARDEN["west"]: LEEUWARDEN["east"], LEEUWARDEN["south"]: LEEUWARDEN["north"]].copy()
    leeuwarden = leeuwarden.drop(columns=columns_2_drop)
    trans = leeuwarden.rename(columns=translation_dict)
    # merge dataframes and filter
    merged = gpd.sjoin(trans, osm_df, how='inner', op='contains')
    columns_to_drop = merged.columns[
        ~merged.columns.isin(trans.columns) & ~merged.columns.isin(['building', 'index_right', 'amenity'])]
    result = merged.drop(columns=columns_to_drop, axis=1)
    mask = result["building"].map(osm_building_types_filter)
    filtered_df = result[mask]
    mask2 = filtered_df["amenity"].isna()
    filtered_2_df = filtered_df[mask2]
    # drop all buildings that have below 45m2 floor area
    larger_buildings = filtered_2_df[filtered_2_df["area"] > 45].reset_index(drop=True)
    # this index is used to kick buildings manually!
    larger_buildings = larger_buildings.rename(columns={"index_right": "OSM_ID"})
    # now kick manual selected buildings:
    complete_filtered = manual_removal_of_large_non_residential_buildings(larger_buildings)
    complete_filtered = complete_filtered.to_crs("epsg:3035")

    complete_filtered.drop(columns=["amenity"]).to_file(Path(r"input_data") / "Leeuwarden.gpkg", driver="GPKG")


def add_location_point_to_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["location"] = gdf.representative_point()
    return gdf


def prepare_data_for_5R1C_transition():
    pass


def load_invert_netherlands_data(invert_city_filter_name: str, country: str, year: int, scen: str):
    df_invert = get_number_of_buildings_from_invert(invert_city_filter_name=invert_city_filter_name,
                                                    country=country,
                                                    year=year,
                                                    scen=scen)

    df_invert.loc[:, "construction_period"] = df_invert.loc[:, "construction_period_start"].astype(str) + "-" + \
                                              df_invert.loc[:, "construction_period_end"].astype(str)

    # drop the rows where buildings are built between 1900-1980 etc. because they refer to
    # commercial buildings and fuck up the following code
    df_invert = df_invert.loc[
                (df_invert.loc[:, "construction_period"] != "1991-1999") &
                (df_invert.loc[:, "construction_period"] != "1975-1990") &
                (df_invert.loc[:, "construction_period"] != "2000-2008"),
                :]
    # remove non residential buildings by filtering for MFH and SFH
    filtered = df_invert[df_invert['name'].str.contains('SFH|MFH')].reset_index(drop=True)
    # correct the construction periods to not leave out random years (eg. 2006 and 2007 is not covered by any period)
    filtered["construction_period"] = filtered["construction_period"].replace({'1880-1974': '1880-1929',
                                                                               '1995-2005': '1995-2007'})
    return filtered


def calculate_5R1C_necessary_parameters(df, year: int, country: str, scen: str):
    # height of the building
    df.loc[:, "height"] = df["b3_h_min"] - df["ground_level"]
    # adjacent area
    df.loc[:, "free wall area (m2)"] = df.loc[:, "free length (m)"] * df.loc[:, "height"]
    # not adjacent area
    df.loc[:, "adjacent area (m2)"] = (df.loc[:, "circumference (m)"] - df.loc[:, "free length (m)"]) * \
                                      df.loc[:, "height"]

    # where there are no estimates from the floors from the TU Delft dataset, we calculate them trough building height
    # and room height
    df["number_of_floors"] = df.apply(lambda row: int(row["floors"]) if not pd.isna(row["floors"]) else np.round(float(row["height"])/float(row["room_height"])), axis=1)
    # if number is 0 after the round, it will be set to 1
    df.loc[df.loc[:, "number_of_floors"] < 1, "number_of_floors"] = 1

    # total wall area
    df.loc[:, "wall area (m2)"] = df.loc[:, "circumference (m)"] * df.loc[:, "height"]
    # ration of adjacent to not adjacent (to compare it to invert later)
    df.loc[:, "percentage attached surface area"] = df.loc[:, "adjacent area (m2)"] / df.loc[:, "wall area (m2)"]
    df_return = get_parameters_from_dynamic_calc_data(df, year, country, scen)

    # demographic information: number of persons per house is number of dwellings (numero_viv) from Urban3R times number
    # of persons per dwelling from invert
    df_return.loc[:, "person_num"] = df_return.loc[:, "number_of_dwellings_per_building"] * df_return.loc[:, "number_of_persons_per_dwelling"]
    # correct the person number because this way it is too high: reduce person number by 1 if the area is below 60m2
    # minimum person numbers in invert is 2...
    df_return.loc[:, "person_num"] = df_return.apply(lambda row: row["person_num"] - 1 if float(row["area"]) < 60 else row["person_num"], axis=1)

    return df_return


def map_invert_data_to_buildings(gdf: gpd.GeoDataFrame,
                                 country: str,
                                 invert_city_filter_name: str,
                                 year: int,
                                 scen: str):
    gdf["original_year_of_construction"] = gdf["original_year_of_construction"].astype(int)
    df_invert = load_invert_netherlands_data(invert_city_filter_name=invert_city_filter_name,
                                             country=country,
                                             year=year,
                                             scen=scen)

    # group invert data after construction period:
    construction_periods = list(df_invert.loc[:, "construction_period"].unique())

    # for some reason, some buildings have a negative building volume. We will exclude them without further investigating
    gdf = gdf.loc[gdf.loc[:, "building_volume"] > 0, :]

    # in Leeuwarden we dont have the classification from the Delft data on the building type (SFH or MFH)
    # therefore we choose this label based on the volume of the building. 450m3 was choosen because then the ration
    # of SFH to MFH resembles aproximately the ration of open source data from: https://allcharts.info/the-netherlands/municipality-leeuwarden/
    # Also the labels from Open Street Maps align well with this approach for the buildings where the labels are available
    gdf.loc[gdf.loc[:, "building_volume"] < 450, "type"] = "SFH"
    gdf.loc[gdf.loc[:, "building_volume"] >= 450, "type"] = "MFH"

    # set construction year to int:
    gdf["original_year_of_construction"] = gdf["original_year_of_construction"].astype(int)


    type_groups = gdf.groupby("type")
    df_list = []
    np.random.seed(42)
    for type, group in type_groups:
        # add the number of stories of the building if not known
        # group["altura_maxima"] = replace_nan_with_distribution(group, "altura_maxima")

        # select a building type from invert based on the construction year and type
        group["invert_construction_period"] = group["original_year_of_construction"].apply(
            lambda x: find_construction_period(construction_periods, x)
        )
        # invert_selection = filter_invert_data_after_type(type=type, invert_df=df_invert)  # does not work for residential (i have a mistake)

        # now select a representative building from invert for each building from Urban3r:
        for i, row in group.iterrows():
            close_selection = df_invert.loc[
                              df_invert["construction_period"] == row['invert_construction_period'], :
                              ]
            # distinguish between SFH and MFH
            sfh_or_mfh = row["type"]

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
            df_list.append(
                pd.DataFrame(pd.concat([selected_building, row.drop(columns=["number_of_buildings"])], axis=0)).T)

    final_df = pd.concat(df_list, axis=0).reset_index(drop=True)

    # we use the minimum building height because we neglect rooms in the attic for the purpose of simlplicity:
    gdf_5r1c = calculate_5R1C_necessary_parameters(final_df, year=year, country=country, scen=scen)

    # with this dataframe we can calculate the new 5R1C parameters:
    new_building_parameters_df, new_total_df = Create5R1CParameters(df=gdf_5r1c).main()
    # give ID Building in to the total df
    new_total_df["ID_Building"] = new_building_parameters_df["ID_Building"]

    # todo save the building df
    # todo update the building dfs for future years
    # save the building df and total df for 2020 once. These dataframes will be reused for the following years:
    new_building_parameters_df.to_excel(
        Path(f"output_data") / f"OperationScenario_Component_Building_{city_name}_non_clustered_{year}_{scen}.xlsx",
        index=False
    )

if __name__ == "__main__":
    Year = 2020
    scenario = "high_eff"
    combined_file_name = Path(r"input_data") / "Leeuwarden.gpkg"
    extended_file_name = Path(r"input_data") / "Leeuwarden_extended.gpkg"
    if not extended_file_name.exists():
        add_osm_information_to_leeuwarden()
        # add the adjacent length and circumference with mosis wonder:
        big_df = calc_premeter(input_lyr=combined_file_name,
                               output_lyr=extended_file_name)


    combined_file = gpd.read_file(extended_file_name)
    df_loc = add_location_point_to_gdf(combined_file)

    map_invert_data_to_buildings(df_loc, invert_city_filter_name="", country="Netherlands", year=Year, scen=scenario)
