from tslearn.metrics import soft_dtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt



def elbowmethod(df:pd.DataFrame):
    """
    Function that finds the best number of clusters and plots a graph of the distortion with the number of clusters
    Args:
        pd.DataFrame: Columns are (country(index), year_week, value)
    Returns:
        int: Best number of cluster 
    """
    import numpy as np
    from tslearn.clustering import silhouette_score
    import matplotlib.pyplot as plt

    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}

    K = range(2,df.shape[0])
    for k in K:
        # Building and fitting the model
        model = TimeSeriesKMeans(n_clusters=k, metric="softdtw", max_iter=50)
        model.fit(df.values[...,np.newaxis])
        distortions.append(silhouette_score(df, model.labels_, metric="softdtw"))

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

    best_num_cluster = np.argmax(distortions)+2
    return best_num_cluster

def construct_model(best_num_cluster:int,df:pd.DataFrame):
    """
    Function that constructs the model, tests the accuracy of the model and produces cluster plots using the model
    Args:
        pd.DataFrame: Columns are (country(index), year_week, value)
        int: Best number of cluster 
    Returns:
        model 

    """
    import numpy as np
    from tslearn.clustering import silhouette_score
    import matplotlib.pyplot as plt

    model = TimeSeriesKMeans(n_clusters=best_num_cluster, metric="softdtw", max_iter=50)

    #### # Test whether the clusters are stable or not 
    model1 = TimeSeriesKMeans(n_clusters=best_num_cluster, metric="softdtw", max_iter=50)
    model1.fit(df.values[...,np.newaxis]) # we need the output to be visualized 

    model2 = TimeSeriesKMeans(n_clusters=best_num_cluster, metric="softdtw", max_iter=50)
    model2.fit(df.values[...,np.newaxis]) # we need the output to be visualized 
    print(normalized_mutual_info_score(model1.labels_, model2.labels_)) # scale the results between 0 (no mutual information) and 1 (perfect correlation)

    ###### Plotting the model ###### 

    model.fit(df.values[...,np.newaxis])
    plt.figure()
    sz = newcases_df.shape[1]
    ylim =newcases_df.values.max()
    for yi in range(best_num_cluster):
        plt.subplot(best_num_cluster, best_num_cluster , yi + 1)
        #for xx in X_train[y_pred == yi]:
            #plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(model.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(0, ylim)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("DTW $k$-means")

    return model 

def countrymatch(df:pd.DataFrame):
    """
    Function that prints the countries in the correponding clusters
    Args:
        pd.DataFrame: Columns are (country(index), year_week, value)

    """
    import numpy as np
    from tslearn.clustering import silhouette_score
    import matplotlib.pyplot as plt

    mapping_country_cluster = {}
    mapping_cluster_country = {}
    for i in range(df.shape[0]):
        mapping_country_cluster[df.index[i]] = model.labels_[i]
        if model.labels_[i] in mapping_cluster_country:
            mapping_cluster_country[model.labels_[i]].append(df.index[i])
        else:
            mapping_cluster_country[model.labels_[i]] = [df.index[i]]
    print(mapping_cluster_country)
    return mapping_cluster_country

