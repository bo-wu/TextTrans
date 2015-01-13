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
    """
    store texture data and return mesh vertex texture colors
    """
    def __init__(self, img_name):
        assert os.path.isfile(img_name)
        self.img = io.imread(img_name)
    
    def get_vertex_color(self, mesh):
        """
        get vertex color from textures
        texture maybe for each triangle, but now treat one-to-one for vertex
        """
        width, height = self.img.shape[:2]

        texture_coord = self.collect_texture_coord(mesh)
        vertex_colors = np.empty((mesh.n_vertices, 3))
        for i in xrange(len(self.texture_coord)):
            vertex_colors[i] = self.img[texture_coord[i, 0]*(width-1),
                                         texture_coord[i, 1]*(height-1), :3]
        return vertex_colors
        
    def collect_texture_coord(self, mesh):
        texture_coord = np.empty((mesh.n_vertices(), 2))
        for vh in mesh.vertices():
            tex2d = mesh.texcoord2D(vh)
            for i in xrange(2):
                texture_coord[vh.idx(), i] = tex2d[i]
        return texture_coord



class TextureTransfer:
    """
    do the main texture transfer algorithm
    """
    def __init__(self, source_mesh_name, img_name, target_mesh_name):
        # load mesh and features
        self.source = MeshFeatures(source_mesh_name)
        source_path = os.path.dirname(source_mesh_name) + '/../features/'
        source_name_ext = os.path.basename(source_mesh_name)
        source_name, ext = os.path.splitext(source_name_ext)
        source_feature = source_path + source_name + '.txt'
        # if true, load features, else compute features
        if os.path.isfile(source_texture):
            self.source.load_features()
        else:
            self.source.assemble_features()

        self.target = MeshFeatures(target_mesh_name)
        target_path = os.path.dirname(target_mesh_name) + '/../features/'
        target_name_ext = os.path.basename(target_mesh_name)
        target_name, ext = os.path.splitext(target_name_ext)
        target_feature = target_path + target_name + '.txt'
        if os.path.isfile(target_feature):
            self.target.load_features()
        else:
            self.target.assemble_features()

        # get source's vertex texture color values
        self.texture = TextureData(img_name)
        self.source_texture = self.texture.get_vertex_color(self.source.mesh)
        
    def feature_texture_cca():
        # use Canonical Correlation Analysis
        # 3 for rgb
        cca = CCA(n_components=3)
        cca.fit(self.source.features, self.source_texture)

if __name__ == '__main__':
    pass


