from pathlib import Path
import geopandas as gpd
from main import LEEUWARDEN, SUCINA, MURCIA, BAARD, KWIDZYN, RUMIA, BASE_EPSG



# load dfs for a city
def load_gdfs_for_city(city: str, folders: list) -> dict:
    dataframes = {}
    for folder in folders:
        file = Path("input_data") / f"{folder}" / f"{city}.gpkg"
        gdf = gpd.read_file(file)
        dataframes[f"{folder}"] = gdf
    return dataframes


def check_if_input_data_exists(folder_names: list):
    city_names = [LEEUWARDEN["city_name"], SUCINA["city_name"],
                  MURCIA["city_name"], BAARD["city_name"],
                  KWIDZYN["city_name"], RUMIA["city_name"]]
    for folder in folder_names:
        path = Path("input_data") / f"{folder}"
        gpkg_file_names = [f.stem for f in path.glob("*.gpkg")]
        if len(gpkg_file_names) == 0:
            print(f"\n NO DATA in {folder} \n")
            continue
        cities_with_data = {city_name for city_name in city_names if any(city_name in file_name for file_name in gpkg_file_names)}
        for city_name in city_names:
            if city_name in cities_with_data:
                print(f"{city_name} has data in {folder}")
            else:
                print(f"{city_name} has NO data in {folder} !!!")


if __name__ == "__main__":
    input_data_folders = ["GHS data", "global_buildings", "OpenStreetMaps", "copernicus_r_3035_10_m_ua-bh-2012_p_2012_v03_r00"]
    check_if_input_data_exists(input_data_folders)


    load_gdfs_for_city(SUCINA["city_name"], input_data_folders)