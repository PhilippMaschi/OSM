import rasterio
import geopandas as gpd
from rasterio.features import shapes
from pathlib import Path
from rasterio.windows import from_bounds
import pyproj
from shapely.geometry import box
from shapely.ops import transform
# from main import LEEUWARDEN, SUCINA, MURCIA, BAARD, KWIDZYN, RUMIA


def list_tif_files(directory: Path):
    """Return a list of all *.tif files in the given directory."""
    dir_path = directory
    return [file for file in dir_path.glob('*.tif')]


def transform_bounds_to_crs(bounds: dict, source_crs: str, target_crs: str) -> dict:
    """Transform the bounds from the source CRS to the target CRS."""

    # Create a shapely geometry for the bounds
    bounding_box = box(bounds["west"], bounds["south"], bounds["east"], bounds["north"])

    # Define the transformation function
    project = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform

    # Apply the transformation
    transformed_box = transform(project, bounding_box)

    # Return the transformed bounds as a dictionary
    return {
        "west": transformed_box.bounds[0],
        "south": transformed_box.bounds[1],
        "east": transformed_box.bounds[2],
        "north": transformed_box.bounds[3]
    }


def is_window_inside_raster(src, window_bounds: dict) -> bool:
    """Check if the window bounds are inside the raster's extent."""
    raster_bounds = src.bounds
    return not (window_bounds["west"] > raster_bounds.right or
                window_bounds["east"] < raster_bounds.left or
                window_bounds["north"] < raster_bounds.bottom or
                window_bounds["south"] > raster_bounds.top)


def load_european_settlement_data(file_path: Path, city_bounds: dict):
    """
    data from https://ghsl.jrc.ec.europa.eu/download.php?ds=builtH
    :return:
    """
    # Read the GeoTIFF file using rasterio
    with rasterio.open(file_path) as src:
        print(f"loading {file_path}")
        raster_crs = src.crs
        valid_bounds = transform_bounds_to_crs(city_bounds, "EPSG:4326", raster_crs)
        # Check if the window is inside the raster's extent. If not skip the city
        if not is_window_inside_raster(src, valid_bounds):
            print(f"Window for city is outside the raster's extent. Skipping extraction for {city_bounds['city_name']}.")
            return gpd.GeoDataFrame()

        # Get the window corresponding to the transformed bounds
        window = from_bounds(valid_bounds["west"], valid_bounds["south"],
                             valid_bounds["east"], valid_bounds["north"], src.transform)

        # Read only the windowed raster data
        image = src.read(window=window)
        mask = image != src.nodata

        # Extract features from the raster data
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
        )
        geoms = list(results)

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
        for city in cities:
            print(f"trying to extract {city['city_name']}")
            gdf = load_european_settlement_data(filepath, city)

            if not gdf.empty:
                gdf.to_file(Path(f"input_data/GHS data") / f"{city['city_name']}.gpkg", driver="GPKG")
                print(f"saved {city} as gpkg file in GHS data")





if __name__ == "__main__":
    # this function takes forever as it iterates through all of the 3 countries and searches for the cities. Only run
    # it when you loose the data on the areas. Data is saved in input_data/GHS data
    BAARD = {"north": 53.1620856552012,
             "south": 53.1314293502190,
             "east": 5.68494449734352,
             "west": 5.64167343762892,
             "city_name": "Baard"}
    main([BAARD])
    # main([LEEUWARDEN, BAARD, MURCIA, SUCINA, KWIDZYN, RUMIA])









