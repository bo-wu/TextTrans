#! /usr/bin/env python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from featureMatch import FeatureMatch

    
class TextureSynthesis:
    """
    synthesis texture from source(example) to target
    """
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
        ######### the main steps ###########
        self.fm = FeatureMatch()
        self.decorrelate_color()
        self.segment_match()
        self.back_to_color()


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
        project back to color from another space(above, scale and rotation)
        target_real_colors.shape is (N, 3)
        """
        target_real_colors = self.back_matrix.dot(self.decorrelated_target.T)
        target_real_colors = target_real_colors.transpose() + self.vertex_colors_mean
        self.target_colors = target_real_colors

        return target_real_colors

    
    def histogram_matching(self, target_part, source_part):
        """
        classical histogram matching
        target_part.shape = (M, 3)
        source_part.shape = (N, 3)
        """
        steps = 301
        temp_target = target_part.copy()
        temp_source = source_part.copy()
        temp_target = temp_target.transpose()
        temp_source = temp_source.transpose()
        target_source = np.hstack((temp_target, temp_source))
        color_min = np.min(target_source, axis=1)
        color_max = np.max(target_source, axis=1)
        target_color_hist = np.empty((temp_target.shape[0], steps-1))
        source_color_hist = np.empty((temp_source.shape[0], steps-1))

        # for each color chanel
        xIndex = np.zeros(temp_target.shape)
        for i in xrange(temp_target.shape[0]):
            bins = np.linspace(color_min[i], color_max[i], steps)
            target_color_hist[i], _ = np.histogram(temp_target[i], bins)
            source_color_hist[i], _ = np.histogram(temp_source[i], bins)
            xIndex[i] = self.fm.pdf_transfer1D(target_color_hist[i], source_color_hist[i])
            scale = (len(xIndex[i]) - 1) / (color_max[i] - color_min[i])
            temp_target[i] = np.interp((temp_target[i] -
                                        color_min[i])*scale,
                                       np.arange(len(xIndex[i])),
                                       xIndex[i]) / scale + color_min[i]

        return xIndex, color_max, color_min


    def histogram_interpolation(self, target_part, index, distance):
        """
        from paper "Texture design using a simplicial complex of morphable textures"
        do the histogram interpolation via xIndex(intensity) interpolation
        target_part.shape = (n, 3)
        index : the index of n nearest source parts of target_part
        """
        xIndices = np.empty((len(index), target_part.shape[1], target_part.shape[0]))
        color_max = np.empty((len(index), target_part.shape[1]))
        color_min = np.empty((len(index), target_part.shape[1]))
        # for each source segment
        for i in xrange(len(index)):
            source_part = self.decorrelated_source[self.source_km.labels_ == index[i]]
            xIndices[i], color_max[i], color_min[i] = self.histogram_matching(target_part, source_part)

        # interpolate xIndex
        distance = 1 / (distance + 1e-10) # small value in case 0
        weight = distance / np.sum(distance)
        xIndex = np.zeros((target_part.shape[1], target_part.shape[0]))
        for i in xrange(len(index)):
            xIndex += weight[i] * xIndices[i]

        # now color_max/min shape is 1x3
        color_max = np.max(color_max, axis=0)
        color_min = np.min(color_min, axis=0)

        temp_target = target_part.copy()
        temp_target = temp_target.transpose()
        # for each color channel
        for i in xrange(target_part.shape[1]):
            scale = (target_part.shape[0]-1) / (color_max[i] - color_min[i])
            temp_target[i] = np.interp((temp_target[i] - color_min[i])*scale,
                                       np.arange(len(xIndex[i])),
                                       xIndex[i])/scale + color_min[i]

        # return the new color
        return temp_target.T



    def segment_match(self, K=10, N=3):
        """
        segment && match
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
        # segment center as data for index, 
        neigh = NearestNeighbors(n_neighbors=N, metric='euclidean')
        neigh.fit(source_km.cluster_centers_)
        distance, index = neigh.kneighbors(target_km.cluster_centers_)

        # for each segment in target, use the N nearest part from source
        # now we are averaging index
        for i in xrange(K):
            target_part_colors = self.decorrelated_target[self.target_km.labels_ == i]
            target_part_colors = self.histogram_interpolation(target_part_colors, index[i], distance[i])
            self.decorrelated_target[self.target_km.labels_ == i] = target_part_colors

    



if __name__ == '__main__':
    pass

