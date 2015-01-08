#! /usr/bin/env python
"""
Author: Bo Wu
contact: wubo.gfkd@gmail.com 
"""

import numpy as np
import openmesh, os, vtk
from helper import veclen
from scipy.special import sph_harm
from scipy import sparse

class MeshFeatures:
    def __init__(self, mesh_name):
        self.mesh = openmesh.TriMesh()
        self.mesh.request_vertex_normals()
        self.mesh.request_vertex_texcoords2D()
        assert os.path.isfile(mesh_name)
        openmesh.read_mesh(self.mesh, mesh_name)
    
    def assemble_features(self):
        for vh in self.mesh.vertices():
            pass

    def calc_normalized_height(self):
        ### I assume y coordinate is height
        self.normalized_height = np.empty(self.mesh.n_vertices())
        for vh in self.mesh.vertices():
            point = self.mesh.point(vh)
            self.normalized_height[vh.idx()] = point[1]

        height_min = np.min(self.normalized_height)
        height_max = np.max(self.normalized_height)
        self.normalized_height = (self.normalized_height - height_min) / (height_max - height_min)


    def collect_vertex_normal(self):
        self.vertex_normal = np.empty((self.mesh.n_vertices(), 3))
        for vh in self.mesh.vertices():
            normal = self.mesh.normal(vh)
            for i in xrange(3):
                self.vertex_normal[vh.idx(), i] = normal[i]


    def calc_mean_curvature(self):
        L, VAI = self.calc_mesh_laplacian()
        HN = -1.0 * VAI.dot(L.dot(self.verts))
        self.mean_curvature = veclen(HN)


    def calc_directional_occlusion(self, phi_sample_num, theta_sample_num):
        phi_table, theta_table = np.mgrid[0:np.pi:phi_sample_num*1j,
                                          0:2*np.pi:theta_sample_num*1j]
        # add some random in the table
        sph_set = np.empty((phi_sample_num, theta_sample_num, 4))      
        sph_set[:,:,0] = sph_harm(0, 0, theta_table, phi_table).real
        sph_set[:,:,1] = sph_harm(-1, 1, theta_table, phi_table).real
        sph_set[:,:,2] = sph_harm(0, 1, theta_table, phi_table).real
        sph_set[:,:,3] = sph_harm(1, 1, theta_table, phi_table).real
        
        # set r large to test intersection
        r = 100.
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        rays = np.empty((phi_sample_num, theta_sample_num, 3))
        rays[:,:,0] = x
        rays[:,:,1] = y
        rays[:,:,2] = z
        
        # prepare the model
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(self.mesh.n_vertices())
        for vh in self.mesh.vertices():
            pos = self.mesh.point(vh)
            points.SetPoint(vh.idx(), pos[0], pos[1], pos[2])

        triangles = vtk.vtkCellArray()
        triangle = vtk.vtkTriangle()
        for fh in self.mesh.faces():
            i = 0
            for fvh in self.mesh.fv(fh):
                triangle.GetPointIds().SetId(i, fvh.idx())
                i = i+1
            triangles.InsertNextCell(triangle)

        poly_mesh = vtk.vtkPolyData()
        poly_mesh.SetPoints(points)
        poly_mesh.SetPolys(triangles)

        bsptree = vtk.vtkModifiedBSPTree()
        bsptree.SetDataSet(poly_mesh)
        bsptree.BuildLocator()

        self.direc_occlu = np.zeros((self.mesh.n_vertices(), 4))
        for vh in self.mesh.vertices():
            pos = self.mesh.point(vh)
            vert = np.array([pos[0], pos[1], pos[2]])
            for i in xrange(phi_sample_num):
                for j in xrange(theta_sample_num):
                    id = bsptree.IntersectWithLine(vert, vert+rays[i,j], 0.01, t,
                                                   intersect_point, pcoord, subid)
                    # id == 0 no intersection
                    if id == 0:
                        self.direc_occlu[vh.idx()] += sph_set[i, j]



    def calc_mesh_laplacian(self):
        """
        computes a sparse matrix representing the discretized laplace-beltrami operator of the mesh
        given by n vertex positions ("verts") and a m triangles ("tris") 
        verts: (n, 3) array (float)
        tris: (m, 3) array (int) - indices into the verts array
        computes the conformal weights ("cotangent weights") for the mesh, ie:
        w_ij = - .5 * (cot \alpha + cot \beta)
        See:
        Olga Sorkine, "Laplacian Mesh Processing"
        and for theoretical comparison of different discretizations, see 
        Max Wardetzky et al., "Discrete Laplace operators: No free lunch"
        returns matrix L that computes the laplacian coordinates, e.g. L * x = delta
        """
        self.verts = np.empty(self.mesh.n_vertices(), 3)
        tris = np.empty(self.mesh.n_faces(), 3)
        for vh in self.mesh.vertices():
            point = self.mesh.point(vh)
            for i in xrange(3):
                self.verts[vh.idx(), i] = point[i]

        for fh in self.mesh.faces():
            i = 0
            for fvh in self.mesh.fv(fn):
                tris[fh.idx(), i] = fvh.idx()
                i = i + 1

        n = len(self.verts)
        W_ij = np.empty(0)
        I = np.empty(0, np.int32)
        J = np.empty(0, np.int32)
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]: # for edge i2 --> i3 facing vertex i1
            vi1 = tris[:,i1] # vertex index of i1
            vi2 = tris[:,i2]
            vi3 = tris[:,i3]
            # vertex vi1 faces the edge between vi2--vi3
            # compute the angle at v1
            # add cotangent angle at v1 to opposite edge v2--v3
            # the cotangent weights are symmetric
            u = self.verts[vi2] - self.verts[vi1]
            v = self.verts[vi3] - self.verts[vi1]
            cotan = (u * v).sum(axis=1) / veclen(np.cross(u, v))
            W_ij = np.append(W_ij, 0.5 * cotan)
            I = np.append(I, vi2)
            J = np.append(J, vi3)
            W_ij = np.append(W_ij, 0.5 * cotan)
            I = np.append(I, vi3)
            J = np.append(J, vi2)
        L = sparse.csr_matrix((W_ij, (I, J)), shape=(n, n))
        # compute diagonal entries
        L = L - sparse.spdiags(L * np.ones(n), 0, n, n)
        L = L.tocsr()
        # area matrix
        e1 = self.verts[tris[:,1]] - self.verts[tris[:,0]]
        e2 = self.verts[tris[:,2]] - self.verts[tris[:,0]]
        n = np.cross(e1, e2)
        triangle_area = .5 * veclen(n)
        # compute per-vertex area
        vertex_area = np.zeros(len(self.verts))
        ta3 = triangle_area / 3
        for i in xrange(tris.shape[1]):
            bc = np.bincount(tris[:,i].astype(int), ta3)
            vertex_area[:len(bc)] += bc
        #VA = sparse.spdiags(vertex_area, 0, len(self.verts), len(self.verts))
        # vertex area inverse
        VAI = sparse.spdiags(1.0/vertex_area, 0, len(self.verts), len(self.verts))
        return L, VAI




if __name__ == '__main__':
    pass

