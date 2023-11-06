import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape
from main import LEEUWARDEN, MURCIA, KWIDZYN, BAARD, RUMIA, SUCINA
from pathlib import Path


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
            new_gdf.to_file(Path("input_data/global_buildings") / f"{city['city_name']}_{row.QuadKey}.geojson",
                            driver="GeoJSON")
            print(f"saved {city['city_name']}")


if __name__ == '__main__':
    # this code takes a while as it downloads all the data and then checks if the region is within the downloaded
    # data.
    load_buildings_from_global_buildings("Spain", MURCIA)
    load_buildings_from_global_buildings("Spain", SUCINA)
    load_buildings_from_global_buildings("Netherlands", LEEUWARDEN)
    load_buildings_from_global_buildings("Netherlands", BAARD)
    load_buildings_from_global_buildings("Poland", KWIDZYN)
    load_buildings_from_global_buildings("Poland", RUMIA)

