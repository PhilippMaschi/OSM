import rasterio
import geopandas as gpd
from rasterio.features import shapes
from pathlib import Path
from rasterio.windows import from_bounds
import pyproj
from shapely.geometry import box
from shapely.ops import transform
from main import LEEUWARDEN, SUCINA, MURCIA, BAARD, KWIDZYN, RUMIA, BASE_EPSG


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
    return transformed_box


def is_window_inside_raster(src, window_box: dict) -> bool:
    """Check if the window bounds are inside the raster's extent."""
    raster_box = box(*src.bounds)
    return raster_box.contains(window_box)


def load_gdf_from_tif_file(file_path: Path, city_bounds: dict):
    """
    data from https://ghsl.jrc.ec.europa.eu/download.php?ds=builtH
        Returns:
    - gdf: GeoDataFrame containing the extracted features
    """
    # Read the GeoTIFF file using rasterio
    with rasterio.open(file_path) as src:
        print(f"loading {file_path}")
        raster_crs = src.crs
        valid_box = transform_bounds_to_crs(city_bounds, BASE_EPSG, raster_crs)
        # Check if the window is inside the raster's extent. If not skip the city
        if not is_window_inside_raster(src, valid_box):
            print(f"Window for city is outside the raster's extent. Skipping extraction for {city_bounds['city_name']}.")
            return gpd.GeoDataFrame()

        # Get the window corresponding to the transformed bounds
        window = from_bounds(*valid_box.bounds, src.transform)
        window_transform = rasterio.windows.transform(window, src.transform)
        # Read only the windowed raster data
        image = src.read(window=window)
        mask = image != src.nodata

        # Extract features from the raster data
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=window_transform))
        )
        geoms = list(results)

    # Convert to a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geoms)
    gdf.crs = raster_crs
    gdf_return = gdf.to_crs("epsg:3035")
    return gdf_return


def global_human_settlement_data(cities: list):
    """
    Load raster data for a specific city's bounds and extract features.
    https://ghsl.jrc.ec.europa.eu/download.php?ds=builtH
    Parameters:
    - cities: Dictionary containing the boundary coordinates of the city
    """
    ghs_files = list_tif_files(Path("input_data/GHS data"))
    # Extract the reprojected bounds
    for filepath in ghs_files:
        for city in cities:
            print(f"trying to extract {city['city_name']}")
            gdf = load_gdf_from_tif_file(filepath, city)

            if not gdf.empty:
                save_gdf_to_gpkg(gdf=gdf, filepath=Path(f"input_data/GHS data") / f"{city['city_name']}.gpkg")
                print(f"saved {city} as gpkg file in GHS data")


def copernicus_data_10m_resolution_building_heights(cities: list):
    input_file = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data\copernicus_r_3035_10_m_ua-bh-2012_p_2012_v03_r00\GTiff\Building_Height_UA2012_WM_10m.tif")
    for city in cities:
        print(f"trying to extract {city['city_name']}")
        gdf = load_gdf_from_tif_file(input_file, city)

        if not gdf.empty:
            save_gdf_to_gpkg(gdf=gdf, filepath=Path(f"input_data/copernicus_r_3035_10_m_ua-bh-2012_p_2012_v03_r00") / f"{city['city_name']}.gpkg")
            print(f"saved {city} as gpkg file in copernicus folder")


def save_gdf_to_gpkg(gdf, filepath: Path):
    """Save a GeoDataFrame to a .gpkg file, overwriting if the file already exists."""
    # Check if the file exists
    if filepath.exists():
        try:
            filepath.unlink()  # remove file
        except:
            print(f"file was not saved because it is used in another programm")

    # Save the GeoDataFrame to .gpkg
    gdf.to_file(filepath, driver="GPKG")


if __name__ == "__main__":
    global_human_settlement_data([KWIDZYN, LEEUWARDEN, BAARD, MURCIA, SUCINA, RUMIA])

    copernicus_data_10m_resolution_building_heights([LEEUWARDEN, BAARD, MURCIA, SUCINA, KWIDZYN, RUMIA])











