import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape
import rasterio
from main import LEEUWARDEN, MURCIA, KWIDZYN, BAARD, RUMIA, SUCINA
from rasterio.features import shapes
from osgeo import gdal
import fiona
from pathlib import Path


def check_if_buildings_are_in_area(gdf: gpd.GeoDataFrame, city: dict):
    # aoi_geom = {
    #     "coordinates": [
    #         [
    #             [LEEUWARDEN["west"], LEEUWARDEN["north"]],
    #             [LEEUWARDEN["west"], LEEUWARDEN["south"]],
    #             [LEEUWARDEN["east"], LEEUWARDEN["north"]],
    #             [LEEUWARDEN["east"], LEEUWARDEN["south"]],
    #         ],
    #     ],
    #     "type": "Polygon",
    #
    # }
    # aoi_shape = shape(aoi_geom)
    # minx, miny, maxx, maxy = aoi_shape.bounds
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
            new_gdf.to_file(Path("input_data/global_buildings") / f"{country}" / f"{city['city_name']}_{row.QuadKey}.geojson",
                        driver="GeoJSON")
            print(f"saved {city['city_name']}")





def read_copernicus():
    tif_path = r'C:\Users\mascherbauer\PycharmProjects\OSM\copernicus_spain_heights.tif'
    ds = gdal.Open(tif_path)
    b = ds.GetRasterBand(1)
    arr = b.ReadAsArray()

    # todo tiff coordinatensystem heraus finden
    # todo zonal statistics um tiff mit murcia df zu "mergen" Rasterstats. Das heißt wir nehmen die Grundfläche von OSM oder URBAN3R
    # um dann diese mit den rasterstats von https://ghsl.jrc.ec.europa.eu/download.php?ds=builtH zu mergen damit wir
    # die Gebäudehöhe erhalten

    # with  gdal.Open(tif_path) as gdal_df:

    # murcia_gdf = gdf.contains()


if __name__ == '__main__':
    load_buildings_from_global_buildings("Spain", SUCINA)
    load_buildings_from_global_buildings("Netherlands", LEEUWARDEN)
    load_buildings_from_global_buildings("Netherlands", BAARD)
    load_buildings_from_global_buildings("Poland", KWIDZYN)
    load_buildings_from_global_buildings("Poland", RUMIA)

    # read_copernicus()
