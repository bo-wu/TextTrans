#! /usr/bin/env python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from featureMatch import FeatureMatch

    
class TextureSynthesis:
    def __init__(self, target_features, target_colors, source_features, source_colors):
        """
        should be Nx3 and Mx3
        target_colors is initial colors
        """
        assert target_features.shape[1] == 3
        assert source_features.shape[1] == 3
        assert source_colors.shape[1] == 3
        self.target_features = target_features
        self.target_colors = target_colors
        self.source_features = source_features
        self.source_colors = source_colors


    def decorrelate_color(self):
        """
        decorrelate_color space into another space
        color.shape is (M, 3)
        apply source color mean and rotation to initial target colors
        """
        self.vertex_colors_mean = np.mean(self.source_colors, axis=0)
        mean_subtract_colors = self.source_colors - self.vertex_colors_mean
        mean_subtract_target = self.target_colors - self.vertex_colors_mean
        u, s, u_t = np.linalg.svd(( mean_subtract_colors.T ).dot(mean_subtract_colors))

        S = np.zeros((3, 3))
        S = np.diag( np.sqrt(s) )
        decorrelated_matrix = (1 / S).dot(u_t)
        self.decorrelated_source = decorrelated_matrix.dot(mean_subtract_colors.T)
        self.decorrelated_source = self.decorrelate_colors.transpose()
        # target colors
        self.decorrelated_target = decorrelated_matrix.dot(mean_subtract_target.T)
        self.decorrelated_target = self.decorrelated_target.transpose()
        self.back_matrix = u.dot(S)


    def back_to_color(self):
        """
        project back to color
        target_colors.shape is (N, 3)
        """
        self.target_real_colors = self.back_matrix.dot(self.decorrelated_target.T)
        self.target_real_colors = self.target_real_colors.transpose() + self.vertex_colors_mean

        return target_real_colors

    
    def histogram_matching(self):
        """
        histogram matching
        decorrelated_target.shape = (M, 3)
        decorrelated_source.shape = (N, 3)
        """
        steps = 301
        temp_target = self.decorrelated_target.T
        temp_source = self.decorrelated_source.T
        target_source = np.hstack((temp_target, temp_source))
        color_min = np.min(target_source, axis=1)
        color_max = np.max(target_source, axis=1)
        target_color_hist = np.empty((temp_target.shape[0], steps-1))
        source_color_hist = np.empty((temp_source.shape[0], steps-1))

        fm = FeatureMatch()
        for i in xrange(self.temp_target.shape[0]):
            bins = np.linspace(color_min[i], color_max[i], steps)
            target_hist[i], _ = np.histogram(temp_target[i], bins)
            source_hist[i], _ = np.histogram(temp_source[i], bins)
            xIndex = fm.pdf_transfer1D(target_hist[i], source_hist[i])
            scale = (len(xIndex) - 1) / (color_max[i] - color_min[i])
            temp_target[i] = np.interp((temp_target[i] -
                                        color_min[i])*scale,
                                       np.arange(len(xIndex)),
                                       xIndex) / scale + color_min[i]

        self.decorrelated_target = temp_target.T
        return temp_target.T

        


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

