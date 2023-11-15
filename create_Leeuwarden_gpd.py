import geopandas as gpd
from pathlib import Path
import pandas as pd
from main import BASE_EPSG, LEEUWARDEN
from tqdm import tqdm
from load_invert_data import get_number_of_buildings_from_invert
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
    'b3_h_maaiveld'
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
    """{
    "identificationInfo": {
    "citation": {
    "title": "3D BAG",
    "date": "2023-10-08",
    "dateType": "creation",
    "edition": "v2023.10.08",
    "identifier": "3652fa0c-6566-11ee-bc83-858db9e376fa"
},"""
    path_2_files = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data\TU_Delft")
    files = list(path_2_files.glob("*.gpkg"))
    big_df = gpd.GeoDataFrame()
    for i, file in tqdm(enumerate(files)):
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

def manual_removal_of_large_non_residential_buildings():
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
    ]


def add_osm_information_to_leeuwarden(delft_df: gpd.GeoDataFrame):
    osm_df = gpd.read_file(Path(r"input_data/OpenStreetMaps") / "Leeuwarden.gpkg")
    osm_df["help_geometry"] = osm_df["geometry"]
    osm_df["geometry"] = osm_df.representative_point()
    osm_df.drop(columns=["help_geometry"]).to_file(Path(r"input_data") / "Leeuwarden_OSM.gpkg", driver="GPKG")
    merged = gpd.sjoin(delft_df, osm_df, how='inner', op='contains')
    columns_to_drop = merged.columns[
        ~merged.columns.isin(delft_df.columns) & ~merged.columns.isin(['building', 'index_right', 'amenity'])]
    result = merged.drop(columns=columns_to_drop, axis=1)
    mask = result["building"].map(osm_building_types_filter)
    filtered_df = result[mask]
    mask2 = filtered_df["amenity"].isna()
    filtered_2_df = filtered_df[mask2]
    # drop all buildings that have below 45m2 floor area
    larger_buildings = filtered_2_df[filtered_2_df["area"] > 45]
    # this index is used to kick buildings manually!
    larger_buildings.rename(columns={"index_right": "OSM_ID"}, inplace=True)
    larger_buildings.drop(columns=["amenity"]).to_file(Path(r"input_data") / "Leeuwarden.gpkg", driver="GPKG")


def add_location_point_to_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["location"] = gdf.representative_point()
    return gdf


def prepare_data_for_5R1C_transition():
    pass


def load_invert_netherlands_data(invert_city_filter_name: str, country: str):
    df_invert = get_number_of_buildings_from_invert(invert_city_filter_name=invert_city_filter_name,
                                                    country=country)

    df_invert.loc[:, "construction_period"] = df_invert.loc[:, "construction_period_start"].astype(str) + "-" + \
                                              df_invert.loc[:, "construction_period_end"].astype(str)

    # drop the rows where buildings are built between 1900-1980 etc. because they refer to
    # commercial buildings and fuck up the following code
    df_invert = df_invert.loc[
                (df_invert.loc[:, "construction_period"] != "1991-1999") &
                (df_invert.loc[:, "construction_period"] != "1975-1990") &
                (df_invert.loc[:, "construction_period"] != "2000-2008"),
                :]
    # Removing b' ' prefix from name
    df_invert['name'] = df_invert['name'].str.replace("b'", '', regex=False)
    df_invert['name'] = df_invert['name'].str.replace("'", '', regex=False)
    # remove non residential buildings by filtering for MFH and SFH
    filtered = df_invert[df_invert['name'].str.contains('SFH|MFH')].reset_index(drop=True)
    # correct the construction periods to not leave out random years (eg. 2006 and 2007 is not covered by any period)
    filtered["construction_period"] = filtered["construction_period"].replace({'1880-1974': '1880-1929',
                                                                               '1995-2005': '1995-2007'})
    return filtered


def map_invert_data_to_buildings(df: gpd.GeoDataFrame, country: str, invert_city_filter_name: str):
    df["original_year_of_construction"] = df["original_year_of_construction"].astype(int)
    df_invert = load_invert_netherlands_data(invert_city_filter_name=invert_city_filter_name, country=country)

    # group invert data after construction period:
    construction_periods = list(df_invert.loc[:, "construction_period"].unique())

    # for each building in the gdf extract the type (SFH or MFH or something else) and the year of construction
    # types = gdf["uso_princi"].unique()
    # print(types)
    # type_groups = gdf.groupby("uso_princi")
    # complete_df = pd.DataFrame()
    # for type, group in type_groups:
    #     # add construction year to the buildings that don't have one
    #     # group["ano_construccion"] = replace_nan_with_distribution(group, "ano_construccion")
    #     group["ano_constr"] = group["ano_constr"].astype(int)
    #     # add the number of stories of the building if not known
    #     # group["altura_maxima"] = replace_nan_with_distribution(group, "altura_maxima")
    #
    #     # select a building type from invert based on the construction year and type
    #     group["invert_construction_period"] = group["ano_constr"].apply(
    #         lambda x: find_construction_period(construction_periods, x)
    #     )
        # invert_selection = filter_invert_data_after_type(type=type, invert_df=df_invert)  # does not work for residential (i have a mistake)

        # now select a representative building from invert for each building from Urban3r:
    #     for i, row in group.iterrows():
    #         close_selection = df_invert.loc[
    #                           df_invert["construction_period"] == row['invert_construction_period'], :
    #                           ]
    #         # distinguish between SFH and MFH
    #         if row["tipologia_"] == "Unifamiliar":
    #             sfh_or_mfh = "SFH"
    #         else:
    #             sfh_or_mfh = "MFH"
    #         mask = close_selection["name"].isin([name for name in close_selection["name"] if sfh_or_mfh in name])
    #         type_selection = close_selection.loc[mask, :]
    #         if type_selection.shape[0] == 0:  # if only SFH or MFH available from invert take this data instead
    #             type_selection = close_selection
    #         # now get select the building after the probability based on the distribution of the number of buildings
    #         # in invert:
    #         distribution = type_selection["number_of_buildings"].astype(float) / \
    #                        type_selection["number_of_buildings"].astype(float).sum()
    #         # draw one sample from the distribution
    #         random_draw = np.random.choice(distribution.index, size=1, p=distribution.values)[0]
    #         selected_building = type_selection.loc[random_draw, :]
    #         # delete number of buildings because this number is 1 for a single building
    #         complete_df = pd.concat([
    #             complete_df,
    #             pd.DataFrame(pd.concat([selected_building, row.drop(columns=["number_of_buildings"])], axis=0)).T
    #         ], axis=0
    #         )
    #
    # final_df = complete_df.reset_index(drop=True)
    # return final_df


if __name__ == "__main__":
    file_name = Path(r"input_data/TU_Delft") / f"Leeuwarden.gpkg"
    if not file_name.exists():
        merge_gpkg_files()
    df = gpd.read_file(file_name)
    add_osm_information_to_leeuwarden(delft_df=df)

    df_loc = add_location_point_to_gdf(df)
    map_invert_data_to_buildings(df_loc, invert_city_filter_name="", country="Netherlands")
