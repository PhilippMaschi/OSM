import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt



def show_invert_hwb_box_plot(city: str):
    df_list = []
    for year in [2020, 2030, 2040, 2050]:
        df = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") /
                           f"{year}_combined_building_df_{city}_non_clustered.xlsx").drop(columns=["rep_point"])
        df.loc[:, "year"] = year

        df_list.append(df)
    big_df = pd.concat(df_list, axis=0)

    sns.boxplot(data=big_df,
                x="year",
                y="hwb",
                hue="type")
    plt.show()

