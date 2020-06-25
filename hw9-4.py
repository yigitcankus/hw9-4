import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import datasets, metrics
from scipy.cluster import hierarchy

heartdisease_df = pd.read_csv("cleveland-0_vs_4.csv")

heartdisease_df = heartdisease_df.replace(to_replace='negative', value=0)
heartdisease_df = heartdisease_df.replace(to_replace='positive', value=1)

heartdisease_df["ca"] = heartdisease_df.ca.replace({'<null>':0})
heartdisease_df["ca"] = heartdisease_df["ca"].astype(np.int64)

heartdisease_df["thal"] = heartdisease_df.thal.replace({'<null>':0})
heartdisease_df["thal"] = heartdisease_df["thal"].astype(np.int64)


X = heartdisease_df.iloc[:, :13]
y = heartdisease_df.iloc[:, 13]


scaler = StandardScaler()
X_std = scaler.fit_transform(X)

linkage_list = ["complete", "average", "ward"]

for link in linkage_list:
    if link !="ward":
        agg_küme = AgglomerativeClustering(linkage=link,
                                              affinity='cosine',
                                              n_clusters=2)
    else:
        agg_küme = AgglomerativeClustering(linkage=link,
                                           affinity='euclidean',
                                           n_clusters=2)
    kümeler = agg_küme.fit_predict(X_std)

    pca = PCA(n_components=2).fit_transform(X_std)

    plt.figure(figsize=(10,5))
    colours = 'rbg'
    for i in range(pca.shape[0]):
        plt.text(pca[i, 0], pca[i, 1], str(kümeler[i]),
                 color=colours[y[i]],
                 fontdict={'weight': 'bold', 'size': 50}
            )

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

    print("Yığınsal Kümeleme Sonuçlarının Ayarlanmış Rand Endeksi : ","(",link," için )",(metrics.adjusted_rand_score(y, kümeler)),)
    print("The silhoutte score of the Agglomerative Clustering solution: {}"
          .format(metrics.silhouette_score(X_std, kümeler, metric='euclidean')))
    print()
    plt.figure(figsize=(20,10))
    hierarchy.dendrogram(hierarchy.linkage(X_std, method='complete'))
    plt.show()




