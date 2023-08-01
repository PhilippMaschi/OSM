import urllib.error
from typing import List, Tuple
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


@dataclass
class Var:
    region = "region"
    year = "year"
    id_hour = "id_hour"
    pv_generation = "pv_generation"
    pv_generation_unit = "pv_generation_unit"
    pv_generation_unit_string = 'W/kW_peak'
    temperature = "temperature"
    temperature_unit = "temperature_unit"
    temperature_unit_string = "°C"
    south = "south"
    east = "east"
    west = "west"
    north = "north"
    radiation_prefix = "radiation_"
    radiation_south = "radiation_south"
    radiation_east = "radiation_east"
    radiation_west = "radiation_west"
    radiation_north = "radiation_north"
    radiation_unit = "radiation_unit"
    radiation_unit_string = "W"


class PVGIS:

    def __init__(self, start_year: int = 2019, end_year: int = 2019):
        self.id_hour = np.arange(1, 8761)
        self.start_year = start_year
        self.end_year = end_year
        self.pv_calculation: int = 1
        self.peak_power: float = 1
        self.pv_loss: int = 14
        self.pv_tech: str = "crystSi"
        self.tracking_type: int = 0
        self.angle: int = 90  # vertical surface
        self.optimal_inclination: int = 1
        self.optimal_angle: int = 1


    """
    pv_calculation: No = 0; Yes = 1
    peak_power: size of PV (in kW_peak)
    pv_loss: system losses in %
    pv_tech: "crystSi", "CIS", "CdTe" and "Unknown".
    tracking_type: type of sun-tracking used,
                    - fixed = 0
                    - single horizontal axis aligned north-south = 1
                    - two-axis tracking = 2
                    - vertical axis tracking = 3
                    - single horizontal axis aligned east-west = 4
                    - single inclined axis aligned north-south = 5
    angle: inclination angle from horizontal plane, which is set to 90° because we are looking at a vertical plane.
    optimal_inclination: Yes = 1, meaning to calculate the optimum inclination angle.
                         All other values (or no value) mean "no". Not relevant for 2-axis tracking.
    optimal_angle: Yes = 1, meaning to calculate the optimum inclination AND orientation angles.
                   All other values (or no value) mean "no". Not relevant for tracking planes.
    """

    @staticmethod
    def scalar2array(value):
        return [value for _ in range(0, 8760)]

    @staticmethod
    def get_url_region_geo_center(nuts_level: int):
        if nuts_level in [0, 1, 2, 3]:
            return f'https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/' \
                   f'NUTS_RG_60M_2021_3035_LEVL_{nuts_level}.geojson'
        else:
            raise Exception(f'Wrong input for NUTS level.')

    def get_geo_center(self, region: str):
        nuts_level = self.get_nuts_level(region)
        nuts = gpd.read_file(self.get_url_region_geo_center(nuts_level))
        transformer = Transformer.from_crs(CRS("EPSG:3035"), CRS("EPSG:4326"))
        point = nuts[nuts.NUTS_ID == region].centroid.values[0]
        lat, lon = transformer.transform(point.y, point.x)
        return lat, lon

    @staticmethod
    def get_nuts_level(region: str):
        return int(len(region) - 2)

    def get_pv_generation(self, region: str = None, lat=None, lon=None) -> np.array:
        """
        either a region or lat and long have to be provided as attributes.
        :param region: the coordinates of the center of the region are used.
        :param lat: region is not used, latitude
        :param lon: region is not used, longitude
        :return: array of 8760 entries with the PV generation of 1 kWp
        """
        if not region:
            if not lat and not lon:
                assert "region or lat and lon have to be provided"
        pv_generation_dict = {}
        self.pv_calculation = 1
        self.optimal_inclination = 1
        self.optimal_angle = 1
        if region:
            lat, lon = self.get_geo_center(region)
        req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}&" \
              f"startyear={self.start_year}&" \
              f"endyear={self.end_year}&" \
              f"pvcalculation={self.pv_calculation}&" \
              f"peakpower={self.peak_power}&" \
              f"loss={self.pv_loss}&" \
              f"pvtechchoice={self.pv_tech}&" \
              f"components={1}&" \
              f"trackingtype={self.tracking_type}&" \
              f"optimalinclination={self.optimal_inclination}&" \
              f"optimalangles={self.optimal_angle}"
        try:
            # Read the csv from api and use 20 columns to receive the source, because depending on the parameters,
            # the number of columns could vary. Empty columns are dropped afterwards:
            df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
            df = df.dropna().reset_index(drop=True)
            # set header to first row
            header = df.iloc[0]
            df = df.iloc[1:, :]
            df.columns = header
            df = df.reset_index(drop=True)
            pv_generation_dict[Var.pv_generation] = pd.to_numeric(df["P"]).to_numpy()  # unit: W
            pv_generation_dict[Var.pv_generation_unit] = self.scalar2array(Var.pv_generation_unit_string)
            return pv_generation_dict
        except urllib.error.HTTPError:
            print(f"pv_generation source is not available for region {region}.")

    def get_temperature_and_solar_radiation(self,
                                            aspect: float,
                                            region: str = None,
                                            lat: float = None,
                                            lon: float = None) -> pd.DataFrame:
        """

        :param aspect: describes the celestial direction: south=0, east=-90, west=90, north=-180
        :param region: name for the region. if region is provided lat and lon will be ignored
        :param lat: latitude, if provided region will be ignored
        :param lon: longitude, if provided region will be ignored
        :return: pandas dataframe with the temperature and solar radiation on north, east, west, south horizontal surface
        """
        if not region:
            if not lat and not lon:
                assert "region or lat and lon have to be provided"
        self.pv_calculation = 0
        self.optimal_inclination = 0
        self.optimal_angle = 0
        if region:
            lat, lon = self.get_geo_center(region)
        req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}&" \
              f"startyear={self.start_year}&" \
              f"endyear={self.end_year}&" \
              f"pvcalculation={self.pv_calculation}&" \
              f"peakpower={self.peak_power}&" \
              f"loss={self.pv_loss}&" \
              f"pvtechchoice={self.pv_tech}&" \
              f"components={1}&" \
              f"trackingtype={self.tracking_type}&" \
              f"optimalinclination={self.optimal_inclination}&" \
              f"optimalangles={self.optimal_angle}&" \
              f"angle={self.angle}&" \
              f"aspect={aspect}"

        # Read the csv from api and use 20 columns to receive the source, because depending on the parameters,
        # the number of columns could vary. Empty columns are dropped afterwards:
        try:
            df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
            df = df.dropna().reset_index(drop=True)
            # set header to first row
            header = df.iloc[0]
            df = df.iloc[1:, :]
            df.columns = header
            return df
        except urllib.error.HTTPError:
            print(f"radiation and temperature is not available for {region} {lat} {lon}.")

    def get_temperature(self,
                        region: str,
                        lat: float = None,
                        lon: float = None) -> dict:
        temperature_dict = {}
        try:
            df = self.get_temperature_and_solar_radiation(region=region,
                                                          aspect=0,
                                                          lat=lat,
                                                          lon=lon)
            temperature_dict[Var.temperature] = pd.to_numeric(df["T2m"].reset_index(drop=True)).values
            temperature_dict[Var.temperature_unit] = self.scalar2array(Var.temperature_unit_string)
            return temperature_dict
        except Exception as e:
            print(f"Temperature source is not available for region {region}.")

    def get_radiation(self,
                      region: str = None,
                      lat: float = None,
                      lon: float = None) -> dict:
        radiation_dict = {}
        celestial_direction_aspect = {
            Var.south: 0,
            Var.east: -90,
            Var.west: 90,
            Var.north: -180
        }
        try:
            for direction, aspect in celestial_direction_aspect.items():
                df = self.get_temperature_and_solar_radiation(region=region,
                                                              aspect=aspect,
                                                              lat=lat,
                                                              lon=lon)
                radiation = pd.to_numeric(df["Gb(i)"]) + pd.to_numeric(df["Gd(i)"])
                radiation_dict[Var.radiation_prefix + direction] = radiation.reset_index(drop=True).to_numpy()
            radiation_dict[Var.radiation_unit] = self.scalar2array(Var.radiation_unit_string)
            return radiation_dict
        except Exception as e:
            print(f"Radiation source is not available for region {region}.")

    def get_pv_gis_data(self,
                        region_name: str,
                        lat: float = None,
                        lon: float = None) -> pd.DataFrame:
        """
        if lat and lon are provided the region name will only be used in the dataframe to describe the data. If lat
        and lon are not used then the regions center will be used as coordinates.
        :param region_name: name of the region or
        :param lat: latitude, if used region_name is only the name
        :param lon: longitude, if used region_name will only be used as name
        :return: pandas dataframe containing the region name, year, hour pv generation (1kWp) temperature and
            radiation on south, north, east and west vertical wall
        """
        result_dict = {
            Var.region: self.scalar2array(region_name),
            Var.year: self.start_year,
            Var.id_hour: self.id_hour,
        }
        # make sure that the region name doesnt overwrite lat and lon if they are provided:
        if lat and lon:
            region = None
        else:
            region = region_name
        pv_generation_dict = self.get_pv_generation(region=region,
                                                    lat=lat,
                                                    lon=lon)
        temperature_dict = self.get_temperature(region=region,
                                                lat=lat,
                                                lon=lon)
        radiation_dict = self.get_radiation(region=region,
                                            lat=lat,
                                            lon=lon)
        try:
            assert pv_generation_dict[Var.pv_generation].sum() != 0
            assert temperature_dict[Var.temperature].sum() != 0
            assert radiation_dict[Var.radiation_south].sum() != 0
            assert radiation_dict[Var.radiation_east].sum() != 0
            assert radiation_dict[Var.radiation_west].sum() != 0
            assert radiation_dict[Var.radiation_north].sum() != 0
            result_dict.update(pv_generation_dict)
            result_dict.update(temperature_dict)
            result_dict.update(radiation_dict)
            result_df = pd.DataFrame.from_dict(result_dict)
            return result_df
        except Exception as e:
            print(f"At least one pv_gis source of Region {region_name} includes all zeros.")

    def download_pv_gis(self,
                        region_names: List[str],
                        coordinates: List[Tuple[float, float]] = None) -> None:
        """
        temperature, radiation and pv generation is downloaded from PV GIS and saved to excel files with the region
        name in them
        :param region_names: list of names of the regions or if no coordinates are provided the center of the regions
                are going to be used as coordinates
        :param coordinates: list of tuples containing (lat, lon) latitude and longitude as floats for each region name
        :return: saves the data to an excel file
        """
        for i, region_name in enumerate(region_names):
            if coordinates:
                lat, lon = coordinates[i]
            else:
                lat, lon = None, None

            print(f'Downloading: {region_name} - lat: {lat} lon: {lon}.')
            region_df = self.get_pv_gis_data(region_name=region_name,
                                             lat=lat,
                                             lon=lon)
            excel_name = f"OperationScenario_RegionWeather_{region_name}.xlsx"
            region_df.to_excel(excel_name)



if __name__ == "__main__":
    # country_list = read_data_excel("NUTS2021")["nuts0"].unique()
    country_list = [
        "Murcia"
    ]
    coordinates = [
        (37.988, -1.124)
    ]
    pv_gis = PVGIS()
    pv_gis.download_pv_gis(region_names=country_list,
                           coordinates=coordinates)


