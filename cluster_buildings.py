import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cophenet, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score
import warnings

# Suppress the FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def hierarchical_cluster(df: pd.DataFrame):
    # possible linkages are: ward, average
    linkage_method = "ward"
    Z = hierarchy.linkage(df, linkage_method)

    # create figure to visualize the cluster
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)
    # Plot with Custom leaves
    dendrogram(Z, leaf_rotation=90, show_contracted=True)  # , annotate_above=0.1)  # , truncate_mode="lastp")

    plt.show()


def kmeans_cluster(df: pd.DataFrame, number_of_cluster: int):
    # define the model
    kmeans_model = KMeans(n_clusters=number_of_cluster, random_state=42)
    # fit the model
    cluster_labels = kmeans_model.fit_predict(df)
    return cluster_labels


def db_scan_cluster_grid_search(df: pd.DataFrame) -> dict:  # doesnt need the number of cluster!
    # TODO find the right method to cluster loads with db scan
    eps_min_sample_dict = {}
    for eps in np.arange(0.006, 0.0075, 0.00001):
        for min_samples in range(2, 10, 1):
            db_model = DBSCAN(eps=eps,
                              min_samples=min_samples,
                              metric="euclidean",
                              metric_params=None,
                              algorithm="auto",
                              p=None,
                              n_jobs=4)
            cluster_labels = db_model.fit(df)
            print(f"{len(np.unique(cluster_labels.labels_))} with eps: {eps} with min samples {min_samples}")
            eps_min_sample_dict[(eps, min_samples)] = len(np.unique(cluster_labels.labels_))
            cluster_sizes = {key: value for key, value in zip(*np.unique(cluster_labels.labels_, return_counts=True))}

    filtered_dict = {key: value for key, value in eps_min_sample_dict.items() if value >= 15}
    return filtered_dict


def db_scan_cluster(df: pd.DataFrame, eps, min_samples):
    db_model = DBSCAN(eps=eps,
                      min_samples=min_samples,
                      metric="euclidean",
                      metric_params=None,
                      algorithm="auto",
                      p=None,
                      n_jobs=4)
    cluster_labels = db_model.fit(df)
    return cluster_labels


def show_heatmap(df, title: str):
    plt.figure(figsize=(10, df.shape[0] / 3))
    # Plot the heatmap
    sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5, cbar=True)
    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Cluster')
    plt.title(f'Cluster Means Heatmap {title}')
    # Show the plot
    plt.close()


def split_df_based_on_Af_quantile(df_q: pd.DataFrame, quantile: float) -> (pd.DataFrame, pd.DataFrame):
    df_upper = df_q.loc[df_q.loc[:, "Af"] > df_q.loc[:, "Af"].quantile(quantile), :]
    df_lower = df_q.loc[df_q.loc[:, "Af"] < df_q.loc[:, "Af"].quantile(quantile), :]
    return df_upper, df_lower


def split_sfh_mfh(df_to_split: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df_to_split["window_area"] = df_to_split["effective_window_area_west_east"] + \
                                 df_to_split["effective_window_area_south"] + \
                                 df_to_split["effective_window_area_north"]
    df_to_split = df_to_split.drop(columns=["internal_gains", "effective_window_area_west_east",
                                            "effective_window_area_south", "effective_window_area_north"])
    df_cluster_sfh = df_to_split.query("type == 'SFH'").drop(columns="type").set_index("ID_Building")
    df_cluster_mfh = df_to_split.query("type == 'MFH'").drop(columns="type").set_index("ID_Building")
    return df_cluster_sfh, df_cluster_mfh


def normalize_df(df_to_norm: pd.DataFrame) -> pd.DataFrame:
    # normalize the dfs:
    # to make sure that all columns contain numbers:
    df_norm = (df_to_norm - df_to_norm.min()) / (df_to_norm.max() - df_to_norm.min())
    return df_norm


def davies_bouldin_analysis(X, k_range: np.array) -> (list, list):
    """
    Calculate the davies bouldin statistic for a given clustering algorithm and dataset.

    Parameters:
    - X: a 2D array of shape (n_samples, n_features) containing the dataset
    - k_range: a range of values for the number of clusters

    Returns:
    - davies bouldin (list): the davies bouldin statistic
    """
    cluster_kmeans = KMeans(random_state=42)
    davies_bouldin_kmeans = []
    for k in k_range:
        cluster_kmeans.set_params(n_clusters=k)
        model_kmeans = cluster_kmeans.fit_predict(X)
        davies_bouldin_kmeans.append(davies_bouldin_score(X, model_kmeans))
    return davies_bouldin_kmeans


def plot_davies_bouldin_index(
        bouldin_list: list,
        k_range: np.array,
        min_davies: int,
        algorithm: str,
        year: int,
        scen: str,
        region: str
) -> None:
    fig = plt.figure()
    plt.plot(k_range, bouldin_list, label="davies bouldin index", marker="D")
    plt.xlabel("k")
    plt.ylabel("davies bouldin value")
    ax = plt.gca()
    low, high = ax.get_ylim()
    plt.vlines(x=min_davies, ymin=low, ymax=high,
               label=f"minimum at k={min_davies}, score={round(np.min(min_davies), 4)}",
               linestyles="--", colors="black")
    plt.legend()
    plt.grid()
    plt.title(f"Davies Bouldin score for {algorithm} Clustering {year} {scen}")
    figure_folder = Path(r"figures") / f"cluster_{region}"
    create_folder(figure_folder)
    plt.savefig(figure_folder / f"Davies_Bouldin_analysis_{algorithm}_{year}_{scen}.png")
    # plt.show()
    plt.close(fig)


def plot_score_results(name: str, x_values: list, y_values: list):
    plt.plot(x_values, y_values, label=name, marker="D")
    plt.xlabel("k")
    plt.ylabel(name)
    plt.vlines(x=x_values[0] + y_values.index(max(y_values)),
               ymin=min(y_values), ymax=max(y_values), colors="r", linestyles="--")
    plt.vlines(x=x_values[0] + y_values.index(min(y_values)),
               ymin=min(y_values), ymax=max(y_values), colors="b", linestyles="--")
    plt.text(s=f"max={round(max(y_values), 2)} at {x_values[0] + y_values.index(max(y_values))} cluster",
             x=x_values[0] + y_values.index(max(y_values)),
             y=max(y_values))
    plt.text(s=f"min={round(min(y_values), 2)} at {x_values[0] + y_values.index(min(y_values))} cluster",
             x=x_values[0] + y_values.index(min(y_values)),
             y=min(y_values))

    plt.grid(which="both")
    plt.title(f"{name}")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\figures") / f"{name}.png")
    plt.close()


def find_number_of_cluster(min_number: int,
                           max_number: int,
                           df_norm: pd.DataFrame,
                           sfh_mfh: str,
                           year: int,
                           scen: str,
                           region: str,
                           ):
    k_range = np.arange(min_number, max_number + 1)
    print(f"analyzing {sfh_mfh} cluster")
    # # Silhouette method for KMeans clustering
    # visualizer_silhouette = KElbowVisualizer(KMeans(), k=k_range, timings=False, metric="silhouette")
    # visualizer_silhouette.fit(df_norm)
    # x_values, y_values = visualizer_silhouette.k_values_, visualizer_silhouette.k_scores_
    # plot_score_results(
    #     name=f"Silhouette method {sfh_mfh}",
    #     x_values=x_values,
    #     y_values=y_values,
    # )
    # print(f"saved optimal number of clusters using KMeans and the silhouette method")

    # Calinski method for KMeans clustering
    # calinski_kmeans = KElbowVisualizer(KMeans(), k=k_range, timings=False, metric="calinski_harabasz")
    # calinski_kmeans.fit(df_norm)
    # x_values, y_values = calinski_kmeans.k_values_, calinski_kmeans.k_scores_
    # plot_score_results(
    #     name=f"Calinski method {sfh_mfh}",
    #     x_values=x_values,
    #     y_values=y_values,
    # )
    # print(f"saved optimal number of clusters using the elbow-calinski KMeans method")

    # calculate the number of clusters with the davies bouldin statistic
    davies_kmeans = davies_bouldin_analysis(X=df_norm, k_range=k_range)
    # optimal number of clusters is the cluster with the lowest boulding index:
    lowest_bouldin_kmeans = np.argmin(davies_kmeans) + min(k_range)
    plot_davies_bouldin_index(bouldin_list=davies_kmeans,
                              k_range=k_range,
                              min_davies=lowest_bouldin_kmeans,
                              algorithm=f"KMeans_{sfh_mfh}",
                              year=year,
                              scen=scen,
                              region=region,
                              )

    print(f"optimal number of cluster using davies bouldin and KMeans {sfh_mfh} {year} {scen}: {lowest_bouldin_kmeans}")
    return lowest_bouldin_kmeans


def calculate_mean(grouped: pd.DataFrame, names_mean: list, weight_column: str):
    if grouped[weight_column].sum() == 0:  # create nan row so it can be easily dropped later
        new_row = pd.Series(data=[np.nan] * len(grouped[names_mean].columns), index=grouped[names_mean].columns)
    else:
        weights = grouped[weight_column] / grouped[weight_column].sum()
        new_row = (grouped[names_mean].T * weights).T.sum()
    return new_row


def create_cluster_dict(dataframe: pd.DataFrame) -> dict:
    """ creates a dictionary containing dfs for SFH and MFH. A seperate df for the 0.9 quantile of
     buildings with highest floor area is created"""
    df_cluster_sfh, df_cluster_mfh = split_sfh_mfh(dataframe)
    shf_upper, sfh_lower = split_df_based_on_Af_quantile(df_cluster_sfh, quantile=0.9)
    mfh_upper, mfh_lower = split_df_based_on_Af_quantile(df_cluster_mfh, quantile=0.9)
    return {
        "sfh_upper": shf_upper,
        "sfh_lower": sfh_lower,
        "mfh_upper": mfh_upper,
        "mfh_lower": mfh_lower
    }


def save_ids_from_each_cluster(counted_ids: dict, region: str, year: int, scen: str) -> None:
    reference_ids_df = pd.DataFrame.from_dict(counted_ids, orient="index").T
    reference_ids_df.to_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") / f"ECEMF_T4.3_{region}_{year}_{scen}" /
                              f"Original_Building_IDs_to_clusters_{region}_{year}.xlsx", index=False)


def create_new_building_df_from_cluster(number_of_cluster: dict,
                                        cluster_dict: dict,
                                        old_df: pd.DataFrame,
                                        region: str,
                                        year: int,
                                        scen: str
                                        ) -> pd.DataFrame:
    """
    create a new building dataframe: instead of 5000 buildings the mean of each cluster
    :param number_of_cluster: dict that contains the number of cluster for each dataset (SFH, MFH)
    :param cluster_dict: dict that contains the name of the cluster and the respective df that should be clustered
    :param old_df: the original df with all buildings
    :param region: str with region name
    :return:
    """

    new_df = pd.DataFrame()
    counter = 1
    count_ids = {}
    for name, n_cluster in number_of_cluster.items():
        # normalize df before clustering
        normalized_df = normalize_df(cluster_dict[name].drop(columns=["person_num"]))
        cluster_labels = kmeans_cluster(normalized_df, n_cluster)

        # add cluster labels to original df
        df_orig = old_df.loc[old_df.loc[:, "ID_Building"].isin(cluster_dict[name].index), :].copy()
        df_orig.loc[:, 'Cluster'] = cluster_labels
        # calculate a small dataframe where the buildings are reduced to
        df_grouped = df_orig.groupby("Cluster")
        # define column names that will be merged by calculating the weighted mean:
        mean_names = [name for name in df_orig.columns if name not in ["ID_Building", "type", "Cluster"]]
        small_df = pd.DataFrame()
        for i, frame in df_grouped:
            new_row = pd.DataFrame(calculate_mean(grouped=frame, names_mean=mean_names, weight_column="Af")).T
            # add number of merged buildings to building df
            new_row.loc[:, "number_of_buildings"] = np.count_nonzero(cluster_labels == i)
            # add the building IDs in case needed later to separate frame
            count_ids[counter] = list(frame.loc[:, "ID_Building"])
            new_row.loc[:, "ID_Building"] = counter
            new_row.loc[:, "type"] = frame["type"].to_numpy()[0]
            counter += 1
            small_df = pd.concat([small_df, new_row])

        new_df = pd.concat([new_df, small_df], axis=0)

        # show the heat maps of the cluster:
        normalized_df.loc[:, 'Cluster'] = cluster_labels
        cluster_means = normalized_df.groupby('Cluster').mean()
        show_heatmap(cluster_means, name)

    new_df.loc[:, "supply_temperature"] = 38
    save_ids_from_each_cluster(counted_ids=count_ids, region=region, year=year, scen=scen)
    return new_df


def load_operation_scenario_ids(city_name: str):
    list_of_ids = [
        "ID_HotWaterTank",
        "ID_SpaceHeatingTank",
        "ID_HeatingElement",
        "ID_Battery",
        "ID_Boiler",
    ]  # space cooling is done in db_init
    pre_name = "OperationScenario_Component_"
    path = Path(r"input_data") / f"FLEX_scenario_files_{city_name}"
    values = {}
    for component_id in list_of_ids:
        xlsx_name = pre_name + component_id.replace("ID_", "") + ".xlsx"
        dataframe = pd.read_excel(path / xlsx_name)
        values[component_id] = dataframe[component_id].to_list()
    return values


def plot_cluster_dict(dict_dfs: dict, region: str):
    plot_list_mfh = []
    plot_list_sfg = []
    for name, dataframe in dict_dfs.items():
        dataframe["type"] = name.upper().replace("_", " ")
        if "mfh" in name.lower():
            plot_list_mfh.append(dataframe.copy())
        else:
            plot_list_sfg.append(dataframe.copy())
        # drop the type column again because otherwise it stays:
        dataframe.drop(columns=["type"], inplace=True)

    plot_df_mfh = pd.concat(plot_list_mfh, axis=0).reset_index(drop=True)
    plot_df_sfh = pd.concat(plot_list_sfg, axis=0).reset_index(drop=True)

    sns.scatterplot(data=plot_df_mfh, x=plot_df_mfh.index, y="Af", hue="type", alpha=0.5)
    plt.title(f"MFH floor area {region}")
    plt.savefig(Path("figures") / "MFH_floor_are.svg")
    plt.close()

    sns.scatterplot(data=plot_df_sfh, x=plot_df_sfh.index, y="Af", hue="type", alpha=0.5)
    plt.title(f"SFH floor area {region}")
    plt.savefig(Path("figures") / "SFH_floor_are.svg")
    plt.close()


def scenario_table_for_flex_model(new_building_df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    create the scenario start file for the flex model
    :param new_building_df: the new_df with the clustered buildings as single buildings
    """
    # create PV table for 5R1C model:
    pv_ids = {
        1: (0, "optimal"),  # kWp, orientation
        2: (5, "optimal"),
        3: (15, "optimal"),
        4: (5, "east"),
        5: (15, "east"),
        6: (5, "west"),
        7: (15, "west"),

    }
    pv_table = new_building_df.loc[:, ["ID_Building", "type"]].copy()
    pv_table_long = pd.DataFrame()
    for building_type, group in pv_table.groupby("type"):
        if building_type == "SFH":
            pv_power = [0, 5]
        else:
            pv_power = [0, 15]
        for pv_id, (kwp, orientation) in pv_ids.items():
            if kwp in pv_power:
                new_frame = group.copy()
                new_frame.loc[:, "ID_PV"] = pv_id
                pv_table_long = pd.concat([pv_table_long, new_frame], axis=0)

    params_values = load_operation_scenario_ids(city_name=region)
    excluded_lists = pv_table_long.loc[:, ["ID_Building", "ID_PV", "type"]].values.tolist()
    first_excluded_key = "ID_Building"
    # create new dictionary where the excluded list is the first element of the first excluded key
    new_dict = {first_excluded_key: excluded_lists}
    for key, values in params_values.items():
        new_dict[key] = values
    keys, values = zip(*new_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    df = pd.DataFrame(permutations_dicts)
    # expand the column with the excluded values with their key names:
    df[["ID_Building", "ID_PV", "type"]] = df[first_excluded_key].apply(pd.Series)

    # now delete the scenarios that can not happen:
    df_start = df.copy()
    # no battery when there is no PV:
    df_start.drop(df_start.loc[
                  (df_start.loc[:, "ID_PV"] == 1) & (df_start.loc[:, "ID_Battery"] != 1), :
                  ].index, inplace=True)
    # correct batteries SFH and MFH
    df_start.drop(df_start.loc[
                  (df_start.loc[:, "type"] == "MFH") & (df_start.loc[:, "ID_Battery"] == 2), :
                  ].index, inplace=True)
    df_start.drop(df_start.loc[
                  (df_start.loc[:, "type"] == "SFH") & (df_start.loc[:, "ID_Battery"] == 3), :
                  ].index, inplace=True)
    # no heating tank when there is no HP
    df_start.drop(df_start.loc[
                  (df_start.loc[:, "ID_Boiler"].isin([1, 4, 5])) & (df_start.loc[:, "ID_SpaceHeatingTank"] != 1), :
                  ].index, inplace=True)
    # big tanks, heating element only for MFH and small only for SFH:
    # Hot water tank
    df_start.drop(df_start.loc[
                  (df_start.loc[:, "type"] == "MFH") & (df_start.loc[:, "ID_HotWaterTank"] == 2), :
                  ].index, inplace=True)
    df_start.drop(df_start.loc[
                  (df_start.loc[:, "type"] == "SFH") & (df_start.loc[:, "ID_HotWaterTank"] == 3), :
                  ].index, inplace=True)
    # space heating tank
    df_start.drop(df_start.loc[
                  (df_start.loc[:, "type"] == "MFH") & (df_start.loc[:, "ID_SpaceHeatingTank"] == 2), :
                  ].index, inplace=True)
    df_start.drop(df_start.loc[
                  (df_start.loc[:, "type"] == "SFH") & (df_start.loc[:, "ID_SpaceHeatingTank"] == 3), :
                  ].index, inplace=True)
    df_start = df_start.reset_index(drop=True)

    # behavior: add ID_Behavior 0 as it is not used and the real IDs are selected in the FLEX model
    df_start["ID_Behavior"] = 0
    df_start.loc[df_start.loc[:, "ID_Boiler"].isin([2, 3, 5]), "ID_Behavior"] = 1  # Air_HP or Ground_HP or Gas
    df_start.loc[df_start.loc[:, "ID_Boiler"] == 1, "ID_Behavior"] = 3  # Electric
    df_start.loc[df_start.loc[:, "ID_Boiler"] == 4, "ID_Behavior"] = 2  # no heating

    final_pv = df_start.copy()

    # create a new building table which corresponds to the PV table (same length, buildings are double)
    merged_df = final_pv.loc[:, ["ID_Building",
                                 "ID_PV",
                                 "ID_Battery",
                                 "ID_HotWaterTank",
                                 "ID_SpaceHeatingTank",
                                 "ID_HeatingElement",
                                 "ID_Boiler",
                                 "ID_Behavior"
                                 ]].copy()
    merged_df = merged_df.merge(new_building_df, on="ID_Building")

    scenario_start = merged_df.loc[:, ["ID_Building",
                                       "ID_PV",
                                       "ID_Battery",
                                       "ID_HotWaterTank",
                                       "ID_SpaceHeatingTank",
                                       "ID_HeatingElement",
                                       "ID_Boiler",
                                       "ID_Behavior"
                                       ]]
    return scenario_start


def plot_year_nr_clusters(plot_dict: dict, scen: str, region: str):
    plot_df = pd.DataFrame.from_dict(
        data=plot_dict, orient="index", columns=["number"]
    ).reset_index().rename(columns={"index": "type"})
    plot_df[['type', 'year']] = plot_df['type'].str.split(' ', expand=True)
    mapping = {"sfh_lower": "SFH below 90th percentile",
               "sfh_upper": "SFH above 90th percentile",
               "mfh_lower": "MFH below 90th percentile",
               "mfh_upper": "MFH above 90th percentile"}
    plot_df["type"] = plot_df["type"].replace(mapping)
    sns.barplot(data=plot_df,
                x="year",
                y="number",
                hue="type")
    plt.ylim(0, 20)
    plt.title(f"Number of clusters for each year and building type in {scen}_{region}")
    plt.savefig(Path(r"figures") / f"number_of_clusters_year_type_{scen}_{region}.svg")
    plt.close()


def create_folder(folder_path: Path):
    if not folder_path.exists():
        folder_path.mkdir(parents=True)


def main(region: str, years: list, scenarios: list):
    year_number_of_cluster = {}
    for scenario in scenarios:
        for year in years:
            df = pd.read_excel(
                Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\output_data") / #f"ECEMF_T4.3_{region}_{year}_{scenario}" /
                f"OperationScenario_Component_Building_{region}_non_clustered_{year}_{scenario}.xlsx"
            )
            cluster_dict = create_cluster_dict(df)
            if year == 2020:
                plot_cluster_dict(cluster_dict, region)
            number_of_cluster = {}  # use davies bouldin as reference
            for sfh_or_mfh, cluster_df in cluster_dict.items():
                number = find_number_of_cluster(min_number=5,
                                                max_number=min([len(cluster_df) - 20, len(cluster_df)//4, 20]),
                                                df_norm=normalize_df(cluster_df.drop(columns=["person_num"])),
                                                sfh_mfh=sfh_or_mfh,
                                                year=year,
                                                scen=scenario,
                                                region=region
                                                )
                number_of_cluster[sfh_or_mfh] = number
                year_number_of_cluster[f"{sfh_or_mfh} {year}"] = number
            for key, dataframe in cluster_dict.items():
                cluster_dict[key] = dataframe.apply(lambda x: pd.to_numeric(x, errors="raise"))
            new_df = create_new_building_df_from_cluster(number_of_cluster=number_of_cluster,
                                                         cluster_dict=cluster_dict,
                                                         old_df=df,
                                                         region=region,
                                                         year=year,
                                                         scen=scenario)
            # save the new building df to the FLEX project:
            output_folder = Path("output_data") / f"ECEMF_T4.3_{region}_{year}_{scenario}"
            create_folder(output_folder)
            new_df.to_excel(output_folder / f"OperationScenario_Component_Building.xlsx",
                            index=False)

            # create the scenario table for the flex model
            start_scenario = scenario_table_for_flex_model(new_building_df=new_df, region=region)
            start_scenario.to_excel(output_folder / f"Scenario_start_{region}_{year}_{scenario}.xlsx", index=False)
        # create plot showing the number of clusters for each category over the years:
        plot_year_nr_clusters(year_number_of_cluster, scenario, region)


if __name__ == "__main__":
    region = "Leeuwarden"
    years = [2020, 2030, 2040, 2050]
    scenarios = ["high_eff", "moderate_eff"]

    main(region, years, scenarios)
