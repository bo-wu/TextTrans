#! /usr/bin/env python
"""
Author: Bo Wu
contact: wubo.gfkd@gmail.com 
"""

from sklearn.cross_decomposition import CCA
import numpy as np
from features import MeshFeatures
from skimage import io, color
import os

class TextureData:
    def __init__(self, img_name):
        assert os.path.isfile(img_name)
        self.img = io.imread(img_name)
    
    def get_mesh_texture_value(self, mesh):
        width, height = self.img.shape[:2]

        self.collect_texture_coord(mesh)
        self.texture_colors = np.empty((mesh.n_vertices, 3))
        for i in xrange(len(self.texture_coord)):
            self.texture_colors[i] = self.img[self.texture_coord[i, 0]*width,
                                              self.texture_coord[i, 1]*height]
        
    def collect_texture_coord(self, mesh):
        self.texture_coord = np.empty((mesh.n_vertices(), 2))
        for vh in mesh.vertices():
            tex2d = mesh.texcoord2D(vh)
            for i in xrange(2):
                self.texture_coord[vh.idx(), i] = tex2d[i]


class TextureTransfer:
    def __init__(self, source_mesh_name, img_name, target_mesh_name):
        self.source = MeshFeatures(source_mesh_name)
        self.target = MeshFeatures(target_mesh_name)
        

    def 

if __name__ == '__main__':
    pass


