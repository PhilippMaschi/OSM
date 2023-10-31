import rasterio
import geopandas as gpd
from rasterio.features import shapes
from pathlib import Path

from main import LEEUWARDEN, SUCINA, MURCIA, BAARD, KWIDZYN, RUMIA


def list_tif_files(directory: Path):
    """Return a list of all *.tif files in the given directory."""
    dir_path = directory
    return [file for file in dir_path.glob('*.tif')]


def load_european_settlement_data(file_path: Path):
    """
    data from https://ghsl.jrc.ec.europa.eu/download.php?ds=builtH
    :return:
    """
    # Read the GeoTIFF file using rasterio
    with rasterio.open(file_path) as src:
        print(f"loading {file_path}")
        # Extract raster data from the window
        image = src.read()  # assuming first band is required
        mask = image != src.nodata
        # Extract features from the raster data
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
        )
        geoms = list(results)
        raster_crs = src.crs

    # Convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geoms)
    gdf.crs = raster_crs
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def main(cities: list):
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
    for filepath in ghs_files:
        gdf = load_european_settlement_data(filepath)
        for city in cities:
            print(f"trying to cut {city}")
            # cut the dataframe to the window:
            gdf_cutted = gdf.cx[
                 city["west"]: city["east"],
                 city["south"]: city["north"]
                 ].copy()
            if not gdf_cutted.empty:
                gdf_cutted.to_file(Path(f"input_data/GHS data") / f"{city['city_name']}.gpkg", driver="GPKG")
                print(f"saved {city} as gpkg file in GHS data")
            else:
                print("cut gdf is empty")




if __name__ == "__main__":
    # this function takes forever as it iterates through all of the 3 countries and searches for the cities. Only run
    # it when you loose the data on the areas. Data is saved in input_data/GHS data
    main([LEEUWARDEN, BAARD, MURCIA, SUCINA, KWIDZYN, RUMIA])

    # gdf_example = load_raster_data(path2file, LOCATIONS["BAARD"])
    # gdf = gpd.read_file(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data") / "leeuwarden_tudelft.gpkg")







