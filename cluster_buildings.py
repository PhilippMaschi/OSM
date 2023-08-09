import pandas as pd
import numpy as np
from pathlib import Path
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
    kmeans_model = KMeans(n_clusters=number_of_cluster)
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
    plt.show()


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
    cluster_kmeans = KMeans()
    davies_bouldin_kmeans = []
    for k in k_range:
        cluster_kmeans.set_params(n_clusters=k)
        model_kmeans = cluster_kmeans.fit_predict(X)
        davies_bouldin_kmeans.append(davies_bouldin_score(X, model_kmeans))
    return davies_bouldin_kmeans


def plot_davies_bouldin_index(bouldin_list: list, k_range: np.array, min_davies: int, algorithm: str) -> None:
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
    plt.title(f"Davies Bouldin score for {algorithm} Clustering")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM\figures") / f"Davies_Bouldin_analysis_{algorithm}.png")
    plt.show()


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
    plt.show()


# hierarchical_cluster(df_cluster_sfh_norm)


def find_number_of_cluster(min_number: int,
                           max_number: int,
                           df_norm: pd.DataFrame,
                           sfh_mfh: str):
    k_range = np.arange(min_number, max_number + 1)
    print(f"analyzing {sfh_mfh} cluster")
    # Silhouette method for KMeans clustering
    visualizer_silhouette = KElbowVisualizer(KMeans(), k=k_range, timings=False, metric="silhouette")
    visualizer_silhouette.fit(df_norm)
    x_values, y_values = visualizer_silhouette.k_values_, visualizer_silhouette.k_scores_
    plot_score_results(
        name=f"Silhouette method {sfh_mfh}",
        x_values=x_values,
        y_values=y_values,
    )
    print(f"saved optimal number of clusters using KMeans and the silhouette method")

    # Calinski method for KMeans clustering
    calinski_kmeans = KElbowVisualizer(KMeans(), k=k_range, timings=False, metric="calinski_harabasz")
    calinski_kmeans.fit(df_norm)
    x_values, y_values = calinski_kmeans.k_values_, calinski_kmeans.k_scores_
    plot_score_results(
        name=f"Calinski method {sfh_mfh}",
        x_values=x_values,
        y_values=y_values,
    )
    print(f"saved optimal number of clusters using the elbow-calinski KMeans method")

    # calculate the number of clusters with the davies bouldin statistic
    davies_kmeans = davies_bouldin_analysis(X=df_norm, k_range=k_range)
    # optimal number of clusters is the cluster with the lowest boulding index:
    lowest_bouldin_kmeans = np.argmin(davies_kmeans) + min(k_range)
    plot_davies_bouldin_index(bouldin_list=davies_kmeans,
                              k_range=k_range,
                              min_davies=lowest_bouldin_kmeans,
                              algorithm=f"KMeans_{sfh_mfh}")

    print(f"optimal number of cluster using davies bouldin and KMeans {sfh_mfh}: {lowest_bouldin_kmeans}")
    return lowest_bouldin_kmeans


def calculate_mean(grouped: pd.DataFrame, names_mean: list, weight_column: str):
    if grouped[weight_column].sum() == 0:  # create nan row so it can be easily dropped later
        new_row = pd.Series(data=[np.nan] * len(grouped[names_mean].columns), index=grouped[names_mean].columns)
    else:
        weights = grouped[weight_column] / grouped[weight_column].sum()
        new_row = (grouped[names_mean].T * weights).T.sum()
    return new_row


if __name__ == "__main__":
    region = "Murcia"
    df = pd.read_excel(Path(r"C:\Users\mascherbauer\PycharmProjects\OSM") / f"OperationScenario_Component_Building_{region}.xlsx")
    df_cluster_sfh, df_cluster_mfh = split_sfh_mfh(df)
    shf_upper, sfh_lower = split_df_based_on_Af_quantile(df_cluster_sfh, quantile=0.9)
    mfh_upper, mfh_lower = split_df_based_on_Af_quantile(df_cluster_mfh, quantile=0.9)

    cluster_dict = {
        "sfh_upper": shf_upper,
        "sfh_lower": sfh_lower,
        "mfh_upper": mfh_upper,
        "mfh_lower": mfh_lower
    }
    number_of_cluster = {}  # use davies bouldin as reference
    for sfh_or_mfh, cluster_df in cluster_dict.items():
        number = find_number_of_cluster(min_number=5,
                                        max_number=min([len(cluster_df) - 20, 30]),
                                        df_norm=normalize_df(cluster_df),
                                        sfh_mfh=sfh_or_mfh)
        number_of_cluster[sfh_or_mfh] = number

    # create a new building dataframe:
    new_df = pd.DataFrame()
    counter = 1
    count_ids = {}
    for name, n_cluster in number_of_cluster.items():
        normalized_df = normalize_df(cluster_dict[name])
        cluster_labels = kmeans_cluster(normalized_df, n_cluster)

        # add cluster labels to original df
        df_orig = df.loc[df.loc[:, "ID_Building"].isin(cluster_dict[name].index), :]
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
    new_df.to_excel(f"OperationScenario_Component_Building_small_{region}.xlsx", index=False)
    reference_ids_df = pd.DataFrame.from_dict(count_ids, orient="index").T
    reference_ids_df.to_excel(f"Original_Building_IDs_to_clusters_{region}.xlsx", index=False)

    # create PV table for 5R1C model:
    pv_ids = {
        1: (0, "optimal"),  # kWp, orientation
        2: (5, "optimal"),
        3: (15, "optimal"),
        4: (0, "east"),
        5: (5, "east"),
        6: (15, "east"),
        7: (0, "west"),
        8: (5, "west"),
        9: (15, "west"),

    }
    pv_table = new_df.loc[:, ["ID_Building", "type"]].copy()
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

    pv_table_long.loc[:, "ID_Battery"] = 1
    pv_table_long.loc[:, "ID_HotWaterTank"] = 1
    pv_table_long.loc[:, "ID_SpaceHeatingTank"] = 1
    pv_table_long.loc[:, "ID_HeatingElement"] = 1

    # split table and add all appliances:
    pv_sfh = pv_table_long.query("type == 'SFH'")
    pv_sfh.loc[:, "ID_HotWaterTank"] = 2
    pv_sfh.loc[:, "ID_SpaceHeatingTank"] = 2
    pv_sfh.loc[:, "ID_HeatingElement"] = 2
    # battery only for buildings with PV
    pv_sfh.loc[pv_sfh.loc[:, "ID_PV"] != 1, "ID_Battery"] = 2
    pv_sfh.loc[pv_sfh.loc[:, "ID_PV"] == 1, "ID_Battery"] = 1

    pv_mfh = pv_table_long.query("type == 'MFH'")
    pv_mfh.loc[:, "ID_HotWaterTank"] = 3
    pv_mfh.loc[:, "ID_SpaceHeatingTank"] = 3
    pv_mfh.loc[:, "ID_HeatingElement"] = 3
    # battery only for buildings with PV
    pv_mfh.loc[pv_mfh.loc[:, "ID_PV"] != 1, "ID_Battery"] = 3
    pv_mfh.loc[pv_mfh.loc[:, "ID_PV"] == 1, "ID_Battery"] = 1

    final_pv = pd.concat([pv_table_long, pv_sfh, pv_mfh], axis=0)

    # create a new building table which corresponds to the PV table (same length, buildings are double)
    merged_df = final_pv.loc[:, ["ID_Building",
                                 "ID_PV",
                                 "ID_Battery",
                                 "ID_HotWaterTank",
                                 "ID_SpaceHeatingTank",
                                 "ID_HeatingElement"]].copy()
    merged_df = merged_df.merge(new_df, on="ID_Building")

    scenario_start = merged_df.loc[:, ["ID_Building",
                                       "ID_PV",
                                       "ID_Battery",
                                       "ID_HotWaterTank",
                                       "ID_SpaceHeatingTank",
                                       "ID_HeatingElement"]]

    scenario_start.to_excel(f"Scenario_start_{region}.xlsx", index=False)

