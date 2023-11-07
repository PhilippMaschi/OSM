import osmnx as ox
from pathlib import Path
from main import LEEUWARDEN, SUCINA, MURCIA, BAARD, KWIDZYN, RUMIA, BASE_EPSG
import geopandas as gpd


def get_osm_gdf(city: dict):
    buildings = ox.geometries_from_bbox(city["north"], city["south"], city["east"], city["west"],
                                        tags={'building': True})
    return buildings


def osm_data(cities: list):
    for city in cities:
        print(f"trying to extract {city['city_name']}")
        gdf = get_osm_gdf(city)
        for column in gdf.columns:
            if any(isinstance(x, list) for x in gdf[column]):
                gdf = gdf.drop(column, axis=1)
        if not gdf.empty:
            save_gdf_to_gpkg(gdf=gdf, filepath=Path("input_data/OpenStreetMaps") / f"{city['city_name']}.gpkg")
            print(f"saved {city['city_name']} as gpkg file in OpenStreetMaps folder")


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


def load_urban3R_and_save():
    file = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\input_data\Urban3R") / "30030.gpkg"
    gdf = gpd.read_file(file)
    if gdf.crs.to_epsg() != BASE_EPSG:
        gdf = gdf.to_crs(BASE_EPSG)
    sucina = gdf.cx[SUCINA["west"]: SUCINA["east"], SUCINA["south"]: SUCINA["north"]].copy()
    murcia = gdf.cx[MURCIA["west"]: MURCIA["east"], MURCIA["south"]: MURCIA["north"]].copy()

    sucina.to_file(Path(r"input_data/Urban3R") / f"Sucina.gpkg", driver="GPKG")
    murcia.to_file(Path(r"input_data/Urban3R") / f"Murcia.gpkg", driver="GPKG")



if __name__ == "__main__":
    cities = [LEEUWARDEN, BAARD, MURCIA, SUCINA, KWIDZYN, RUMIA]
    load_urban3R_and_save()
    osm_data(cities)






