import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from rasterio.windows import from_bounds
from pathlib import Path
import pyproj
from shapely.ops import transform
from shapely.geometry import box

# Define locations in a dictionary for better management
LOCATIONS = {
    "LEEUWARDEN": {
        "north": 53.2178674080337,
        "south": 53.1932515262881,
        "east": 5.82625369878255,
        "west": 5.76735091362368,
        "city_name": "Leeuwarden"
    },
    "BAARD": {
        "north": 53.1620856552012,
        "south": 53.1314293502190,
        "east": 5.68494449734352,
        "west": 5.64167343762892,
        "city_name": "Baard"
    }
}


def load_raster_data(filepath, city_bounds):
    """
    Load raster data for a specific city's bounds and extract features.

    Parameters:
    - filepath: Path to the raster file
    - city_bounds: Dictionary containing the boundary coordinates of the city

    Returns:
    - gdf: GeoDataFrame containing the extracted features
    """
    # Create bounding box
    bbox = box(
        minx=city_bounds["west"],
        miny=city_bounds["south"],
        maxx=city_bounds["east"],
        maxy=city_bounds["north"]
    )

    # Define the source (WGS 84) and target (Mollweide) CRS
    src_crs = pyproj.CRS("EPSG:4326")  # WGS 84
    dst_crs = pyproj.CRS("ESRI:54009")  # Mollweide

    # Define the transformation function
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform

    # Reproject the bounding box
    reprojected_bbox = transform(project, bbox)

    # Extract the reprojected bounds
    west, south, east, north = reprojected_bbox.bounds

    with rasterio.open(filepath) as src:
        # Calculate the corresponding window for the raster using the reprojected bounds
        window = from_bounds(
            left=west,
            bottom=south,
            right=east,
            top=north,
            transform=src.transform
        )
        # todo window doesnt work, load the whole dataframe and cx_ it later
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
    # cut the dataframe to the window:
    gdf_cutted = gdf.cx[
         LOCATIONS["LEEUWARDEN"]["west"]: LOCATIONS["LEEUWARDEN"]["east"],
         LOCATIONS["LEEUWARDEN"]["south"]: LOCATIONS["LEEUWARDEN"]["north"]
         ].copy()
    return gdf_cutted




if __name__ == "__main__":
    path2file = Path(
        r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data\GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R3_C19\GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R3_C19.tif"
    )
    gdf_example = load_raster_data(path2file, LOCATIONS["LEEUWARDEN"])
    gdf_example.to_file("Leuwarden_heights.shp")
    # Printing the first few rows of the GeoDataFrame for verification
    gdf_example.head()






