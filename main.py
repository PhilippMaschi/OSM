import osmapi
import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

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

def show_place(north: float, south: float, east: float, west: float):
    G = ox.graph_from_bbox(north, south, east, west, network_type='all')
    # plot the network graph
    fig, ax = ox.plot_graph(G)


def show_number_of_buildings(city: dict):
    # extract the buildings within the bounding box
    buildings = ox.geometries_from_bbox(city["north"], city["south"], city["east"], city["west"],
                                        tags={'building': True})

    # extract the building types from the buildingsimport matplotlib.pyplot as plt
    building_types = list(set(buildings['building'].values))
    print(f"totoal number of buildings in {city['city_name']}: {len(buildings)}")
    relevant_building_types = ["apartments", "residential", "industrial", "office", "house", "hotel"]
    for type in relevant_building_types:
        print(f"number of buildings of type {type}: {len(buildings.query(f'building==@type'))}")

    print("\n \n \n")
    return building_types


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Leeuwarden
show_number_of_buildings(MURCIA)

