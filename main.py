import osmapi
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import plotly.express as px

LEEUWARDEN = {"north": 53.2178674080337,
              "south": 53.1932515262881,
              "east": 5.82625369878255,
              "west": 5.76735091362368,
              "city_name": "Leeuwarden"}

KWIDZYN = {"north": 53.7487987337842,
           "south": 53.7164945915439,
           "east": 18.9535734503598,
           "west": 18.9095466059208,
           "city_name": "Kwidzyn"}

BAARD = {"north": 53.1620856552012,
         "south": 53.1314293502190,
         "east": 5.68494449734352,
         "west": 5.64167343762892,
         "city_name": "Baard"}

RUMIA = {"north": 54.5877326995821,
         "south": 54.5560197018211,
         "east": 18.4243116632352,
         "west": 18.3784147154927,
         "city_name": "Rumia"}

SUCINA = {"north": 37.9007686931277,
          "south": 37.8459872260350,
          "east": -0.929828651725462,
          "west": -0.976709171036036,
          "city_name": "Sucina"}

MURCIA = {"north": 37.9988137604873,
          "south": 37.9656536677982,
          "east": -1.10710096876283,
          "west": -1.13912542769010,
          "city_name": "Murcia"}

TRANSLATION_DICT = {
    "Almacenaje": "storage",
    "Aparcamiento": "parking",
    "Residencial": "residential",
    "Público": "public",
    "Piscinas": "swimming pool",
    "EnseñanzaCultural": "cultural education",
    "HoteleroRestauración": "hotel",
    "IndustrialResto": "industrial site",
    "OtrosNoCal": "other no cal",
    "OtrosCal": "other cal",
    "Industrial": "industrial",
    "Comercial": "commercial",
    "VincViv": "social housing",
    "Deportivo": "sport",
    "Común": "shared",
    "Oficinas": "offices",
}

def show_place(north: float, south: float, east: float, west: float):
    G = ox.graph_from_bbox(north, south, east, west, network_type='all')
    # plot the network graph
    fig, ax = ox.plot_graph(G)


def show_number_of_buildings(city: dict, big_df: pd.DataFrame):
    # extract the buildings within the bounding box
    buildings = ox.geometries_from_bbox(city["north"], city["south"], city["east"], city["west"],
                                        tags={'building': True})
    new_df = pd.DataFrame(buildings['building']).reset_index(drop=True)
    new_df["city"] = city["city_name"]
    df = pd.concat([big_df, new_df], axis=0)
    # extract the building types from the buildings
    print(f"totoal number of buildings in {city['city_name']}: {len(buildings)} \n")
    return df


def show_murcia_data():
    murcia_df = gpd.read_file("30030.gpkg")
    # create a Shapely box object from the bounding box coordinates
    bbox = box(MURCIA["west"], MURCIA["south"], MURCIA["east"], MURCIA["north"])
    filtered_gdf = murcia_df.cx[MURCIA["west"]: MURCIA["east"],  MURCIA["south"]: MURCIA["north"]].copy()
    filtered_gdf["number_of_buildings"] = 1
    filtered_gdf["uso_principal"] = filtered_gdf["uso_principal"].replace(TRANSLATION_DICT)
    murcia_numbers = filtered_gdf.groupby("uso_principal")["number_of_buildings"].sum().reset_index()


    # get OSM murcia data
    osm_df = pd.DataFrame(columns=["building", "city"])
    osm_df = show_number_of_buildings(MURCIA, osm_df)
    osm_df["number_of_buildings"] = 1
    osm_numbers = osm_df.groupby("building")["number_of_buildings"].sum().reset_index()


    common_names = set(osm_df['building']).intersection(set(murcia_numbers['uso_principal']))

    murcia = murcia_numbers.query(f"uso_principal in {list(common_names)}")
    # define the percentage of the different buildings:
    murcia["percentage"] = murcia["number_of_buildings"] / murcia["number_of_buildings"].sum()
    murcia = murcia.reset_index(drop=True)

    # add the buildings that are configured as "yes" to the number of buildings based on the percentage of murcia buildings
    osm = osm_numbers.query(f"building in {list(common_names)}").reset_index(drop=True)
    osm["number_of_buildings"] = osm["number_of_buildings"] + float(osm_numbers.query("building == 'yes'")["number_of_buildings"]) * murcia["percentage"]

    murcia["source"] = "urban3r"
    osm["source"] = "osm"
    df = pd.concat([murcia.drop(columns=["percentage"]).rename(columns={"uso_principal": "building"}), osm], axis=0)

    fig = px.bar(
        data_frame=df,
        x="building",
        y="number_of_buildings",
        color="source",
        barmode="group",
    )
    fig.show()




def plotly_number_of_buildings(long_df: pd.DataFrame):
    fig = px.bar(
        data_frame=long_df,
        x="building_type",
        y="count",
        color="city",
        barmode="group"
    )
    fig.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    show_murcia_data()

    city_list = [MURCIA, KWIDZYN, LEEUWARDEN, BAARD, SUCINA, RUMIA]
    big_df = pd.DataFrame(columns=["building", "city"])
    for city in city_list:
        big_df = show_number_of_buildings(city, big_df)

    counts = big_df.groupby("city")['building'].value_counts()
    long_df = pd.DataFrame({'building_type': counts.index.get_level_values("building"),
                           'city': counts.index.get_level_values("city"),
                           'count': counts.values})
    long_df["count"] = long_df["count"].astype(float)
    plotly_number_of_buildings(long_df)




