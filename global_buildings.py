import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape
from main import LEEUWARDEN, MURCIA, KWIDZYN, BAARD, RUMIA, SUCINA
from pathlib import Path
from collections import Counter

def check_if_buildings_are_in_area(gdf: gpd.GeoDataFrame, city: dict):
    bbox = box(city["west"], city["south"], city["east"], city["north"])
    any_within = gdf["geometry"].within(bbox).any()
    if any_within:
        return True
    else:
        return False


def load_buildings_from_global_buildings(country: str, city: dict):
    """
    This snippet demonstrates how to access and convert the buildings
    data from .csv.gz to geojson for use in common GIS tools. You will
    need to install pandas, geopandas, and shapely.

    https://github.com/microsoft/GlobalMLBuildingFootprints

    """
    dataset_links = pd.read_csv("https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv")
    links = dataset_links[dataset_links.Location == country]
    for _, row in links.iterrows():
        df = pd.read_json(row.Url, lines=True)
        df['geometry'] = df['geometry'].apply(shape)
        gdf = gpd.GeoDataFrame(df, crs=4326)
        within = check_if_buildings_are_in_area(gdf, city)
        print(f"{row.QuadKey}: {within}")
        if within:
            gdf_cutted = gdf.cx[city["west"]: city["east"], city["south"]: city["north"]].copy()
            new_gdf = gdf_cutted.to_crs("epsg:3035")
            new_gdf.to_file(Path("input_data/global_buildings") / f"{city['city_name']}_{row.QuadKey}.gpkg",
                            driver="GPKG")
            print(f"saved {city['city_name']}")


def delete_numbers_in_file_names():
    # if a city has two files we need to merge the files before renaming:
    paths = [file for file in Path("input_data/global_buildings").glob("*.gpkg")]
    remove_digits = str.maketrans('', '', '0123456789')
    file_names = [f.stem.translate(remove_digits).replace('_', '') for f in paths]
    if len(file_names) != len(set(file_names)):
        # doubles exist and we have to merge the dataframes
        counts = Counter(file_names)
        duplicates = [item for item, count in counts.items() if count > 1]
        gdfs = []
        for duplicate in duplicates:
            double_files = [file for file in paths if duplicate in file.stem]
            # load the dataframes and merge them
            for f in double_files:
                gdfs.append(gpd.read_file(f))
                f.unlink()

        gdf_concatenated = pd.concat(gdfs, ignore_index=True)
        gdf_concatenated.to_file(Path(r"input_data/global_buildings") / f"{duplicates[0]}.gpkg", driver="GPKG")


    new_paths = [file for file in Path("input_data/global_buildings").glob("*.gpkg")]
    for file in new_paths:
        new_name = f"{file.stem.translate(remove_digits).replace('_', '')}.gpkg"
        file.rename(file.parent / new_name)

if __name__ == '__main__':
    # this code takes a while as it downloads all the data and then checks if the region is within the downloaded
    # data.
    load_buildings_from_global_buildings("Spain", MURCIA)
    load_buildings_from_global_buildings("Spain", SUCINA)
    load_buildings_from_global_buildings("Netherlands", LEEUWARDEN)
    load_buildings_from_global_buildings("Netherlands", BAARD)
    load_buildings_from_global_buildings("Poland", KWIDZYN)
    load_buildings_from_global_buildings("Poland", RUMIA)

    delete_numbers_in_file_names()

