#! /usr/bin/env python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

    
class TextureSynthesis:
    def __init__(self, target_features, source_features, source_colors):
        """
        this one should be Nx3
        """
        assert target_features.shape[1] == 3
        assert source_features.shape[1] == 3
        assert source_colors.shape[1] == 3
        self.target_features = target_features
        self.source_features = source_features
        self.source_colors = source_colors


    def decorrelate_color(self):
        """
        decorrelate_color space into another space
        """
        self.vertex_colors_mean = np.mean(self.source_colors, axis=0)
        mean_subtract_colors = self.source_colors - self.vertex_colors_mean
        self.decorelated_colors = np.linalg.svd()



    def segment_match(self, K=10, N=3):
        """
        segment both target_features and source_features to K parts, 
        then for each part in target choose n nearest parts from source to 
        interpolate as target
        target_features: (n, m) # n samples, and m features
        source_features: (l, m)
        """
        # Kmeans segment
        source_km = KMeans(n_clusters=K, init='k-means++', max_iter=100,
                           n_init=10, n_jobs=-3)   
        source_km.fit(self.source_features)
        target_km = KMeans(n_clusters=K, init='k-means++', max_iter=100,
                           n_init=10, n_jobs=-3)
        target_km.fit(self.target_features)

        # nearest neighbors
        neigh = NearestNeighbors(n_neighbors=N, metric='euclidean')
        neigh.fit(source_km.cluster_centers_)
        distance, index = neigh.kneighbors(target_km.cluster_centers_)

        fm = FeatureMatch(iter=10)
        # for each segment in target, use the N nearest part from source
        for i in xrange(K):
            target_segment_features = self.target_features[self.target_km.labels == i]
            for j in xrange(N):
                # segment center as data for index, 
                source_segment_features = self.source_features[self.source_km.labels==index[i, j]]
                # the idea below is averaging the histogram
                #fm.match(target_segment_features.T, source_segment_features.T)
            # now we are averaging index



if __name__ == '__main__':
    pass

