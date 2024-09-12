# Copyright 2024 Takuya Fujimura

import logging

import numpy as np
from sklearn.cluster import KMeans

from .utils import get_embed_from_df

# from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def get_embed_from_df(df):
    e_cols = [col for col in df.keys() if col[0] == "e" and int(col[1:]) >= 0]
    return df[e_cols].values


class TargetKmeans:
    def __init__(self, hp):
        self.hp = hp
        self.kmeans = KMeans(n_clusters=hp, random_state=0)

    def fit(self, train_df):
        is_target = train_df["is_target"].values
        embed = get_embed_from_df(train_df)
        self.kmeans.fit(embed[is_target == 0])
        self.means_ta = embed[is_target == 1]

        if sum(is_target) == 0:
            logging.warn("TargetKNN.fit: number of target data is 0.")
            return False
        return True

    def anomaly_score(self, test_df):
        pass


class TargetKmeansCos(TargetKmeans):
    def __init__(self, hp):
        super().__init__(hp)

    def anomaly_score(self, test_df):
        embed = get_embed_from_df(test_df)
        # compute cosine distances
        means_so = self.kmeans.cluster_centers_
        eval_cos = np.min(
            1 - np.dot(embed, self.means_ta.transpose()),
            axis=-1,
            keepdims=True,
        )
        eval_cos = np.minimum(
            eval_cos,
            np.min(
                1 - np.dot(embed, means_so.transpose()),
                axis=-1,
                keepdims=True,
            ),
        )
        return {"plain": eval_cos.squeeze(-1)}


class TargetKmeansEuclid(TargetKmeans):
    def __init__(self, hp):
        super().__init__(hp)

    def anomaly_score(self, test_df):
        embed = get_embed_from_df(test_df)
        # compute cosine distances
        means_so = self.kmeans.cluster_centers_
        anom_score_ta = np.zeros((len(self.means_ta), len(embed)))
        for i in range(len(self.means_ta)):
            anom_score_ta[i] = np.sqrt(np.sum((embed - self.means_ta[i]) ** 2, axis=-1))
        anom_score_ta = np.min(anom_score_ta, axis=0)
        anom_score_so = np.zeros((self.hp, len(embed)))
        for i in range(self.hp):
            anom_score_so[i] = np.sqrt(np.sum((embed - means_so[i]) ** 2, axis=-1))
        anom_score_so = np.min(anom_score_so, axis=0)
        return {"plain": np.minimum(anom_score_so, anom_score_ta)}
