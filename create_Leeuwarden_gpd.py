import geopandas as gpd
from pathlib import Path
import pandas as pd
from main import BASE_EPSG, LEEUWARDEN
from tqdm import tqdm

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
]

filter_dict = {
    'status': ['Pand in gebruik', 'Verbouwing pand'],
}

translation_dict = {
    "b3_dak_type": "roof_type",
    'b3_h_maaiveld': "heigth",
    'b3_kas_warenhuis': "warehouse",
    'b3_opp_buitenmuur': "outer_wall_area",
    'b3_opp_dak_plat': "flat_roof_area",
    'b3_opp_dak_schuin': "sloping_roof_area",
    'b3_opp_grond': "ground_area",
    'b3_opp_scheidingsmuur': "partition_wall_area",
    'b3_reconstructie_onvolledig': "incomplete_reconstruction",
    'b3_volume_lod22': "building_volume",
    'oorspronkelijkbouwjaar': "original_year_of_construction",
    'voorkomenidentificatie': "occurrence_identification",
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

        big_df = pd.concat([big_df, filtered_df], axis=0)

    new_df = big_df.to_crs(BASE_EPSG)
    leeuwarden_df = new_df.cx[LEEUWARDEN["west"]: LEEUWARDEN["east"], LEEUWARDEN["south"]: LEEUWARDEN["north"]]
    leeuwarden_df.to_file(Path(r"input_data/TU_Delft") / f"Leeuwarden.gpkg", driver="GPKG")


if __name__ == "__main__":
    merge_gpkg_files()
