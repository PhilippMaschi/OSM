import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from rasterio.windows import from_bounds
from pathlib import Path
import pyproj
from shapely.ops import transform
from shapely.geometry import box, shape
from main import LEEUWARDEN, SUCINA, MURCIA, BAARD, KWIDZYN, RUMIA


def list_tif_files(directory: Path):
    """Return a list of all *.tif files in the given directory."""
    dir_path = directory
    return [file for file in dir_path.glob('*.tif')]

def copernicus():
    """
    data from https://ghsl.jrc.ec.europa.eu/download.php?ds=builtH
    :return:
    """

    # Path to the GeoTIFF file
    tif_path = r'C:\Users\mascherbauer\PycharmProjects\OSM\copernicus_spain_heights.tif'

    # Read the GeoTIFF file using rasterio
    with rasterio.open(tif_path) as src:
        # Read the raster data as a numpy array
        raster_data = src.read(1)  # Assuming a single band GeoTIFF

        # Retrieve the spatial information (e.g., CRS and transform)
        crs = src.crs
        transform = src.transform

    # Create a list to store the polygon geometries
    polygons = []

    # Generate polygons from the raster data
    i = 0
    for geom, value in shapes(raster_data, transform=transform):
        if value != 0:  # Skip empty polygons
            polygons.append(shape(geom))
            print(i)
            i += 1

    # Create a GeoDataFrame from the polygons
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    # Print the GeoDataFrame
    print(gdf.head())
    gdf.to_file(f"copernicus.geojson", driver="GeoJSON")

def load_european_settlement_data(cities: list):
    """
    Load raster data for a specific city's bounds and extract features.
    https://ghsl.jrc.ec.europa.eu/download.php?ds=builtH
    Parameters:
    - filepath: Path to the raster file
    - city_bounds: Dictionary containing the boundary coordinates of the city

    Returns:
    - gdf: GeoDataFrame containing the extracted features
    """
    ghs_files = list_tif_files(Path("input_data/GHS data"))
    # Extract the reprojected bounds
    big_gdfs = []
    for filepath in ghs_files:
        with rasterio.open(filepath) as src:
            print(f"loading {filepath}")
            # Extract raster data from the window
            image = src.read()  # assuming first band is required

            mask = image != src.nodata

            # Extract features from the raster data
            results = (
                {'properties': {'value': v}, 'geometry': s}
                for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
            )

            geoms = list(results)

        # Convert to a GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geoms)
        gdf.crs = "ESRI:54009"
        gdf = gdf.to_crs("EPSG:4326")

        big_gdfs.append(gdf)

    for big_gdf in big_gdfs:
        for city in cities:
            print(f"trying to cut {city}")
            # cut the dataframe to the window:
            gdf_cutted = big_gdf.cx[
                 city["west"]: city["east"],
                 city["south"]: city["north"]
                 ].copy()
            if gdf_cutted.empty:
                continue
            else:
                gdf_cutted.to_file(Path(f"input_data/GHS data") / f"{city}.gpkg", driver="GPKG")
                print(f"saved {city} as gpkg file in GHS data")




if __name__ == "__main__":
    # this function takes forever as it iterates through all of the 3 countries and searches for the cities. Only run
    # it when you loose the data on the areas. Data is saved in input_data/GHS data
    load_european_settlement_data([LEEUWARDEN, BAARD, MURCIA, SUCINA, KWIDZYN, RUMIA])

    # gdf_example = load_raster_data(path2file, LOCATIONS["BAARD"])
    # gdf = gpd.read_file(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data") / "leeuwarden_tudelft.gpkg")







