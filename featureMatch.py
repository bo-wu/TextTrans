#! /usr/bin/env python
import numpy as np
import sys

class FeatureMatch:
    def __init__(self, iter):
        B = np.array([1,0,0, 0,1,0, 0,0,1, 2/3.,2/3.,-1/3.,
                           2/3.,-1/3.,2/3., -1/3.,2/3.,2/3.])
        B.shape = (6,3)
        self.itera = iter
        self.R = np.empty((6, 3, iter))
        self.R[:,:,0]= B
        for i in xrange(1, iter):
            #generate rotation
            a = np.random.randn(3, 3)
            q, r = np.linalg.qr(a)
            self.R[:,:,i] = B.dot(q)
    

    def pdf_transfer1D(self, pX, pY):
        """
        1d pdf transfer
        """
        nbins = pX.shape[0]
        PX = np.cumsum(pX)
        PX = PX / PX[-1]
        PY = np.cumsum(pY)
        PY = PY / PY[-1]
        #
        small_damping = np.arange(nbins+2, dtype=np.float) / (nbins+1) * 1e-3 
        PX = np.insert(PX, 0, 0)
        PX = np.append(PX, nbins)
        PX = PX + small_damping
        PY = np.insert(PY, 0, 0)
        PY = np.append(PY, nbins)
        PY = PY + small_damping
        
        yInterval = np.arange(nbins, dtype=np.float)
        yInterval = yInterval + 1e-16
        yInterval = np.insert(yInterval, 0, 0)
        yInterval = np.append(yInterval, nbins+1e-10)
        xInterval = np.interp(PX, PY, yInterval)
        return xInterval[1:-1]



    def match(self, target_features, source_features):
        """
        here target should match source feature distribution
        target_features.shape = (3, m)
        source_features.shape = (3, n)
        """
        if source_features.shape[0] != 3 or target_features.shape[0] != 3:
            print 'features shape should be nx3, match failure'
            sys.exit()
        else:
            steps = 301
            for i in xrange(self.itera):
                Rotate = self.R[:,:,i]
                temp_target = Rotate.dot(target_features)
                temp_source = Rotate.dot(source_features)
                target_source = np.hstack((temp_target, temp_source))
                feature_min = np.min(target_source, axis=1)
                feature_max = np.max(target_source, axis=1)
                target_hist = np.empty((temp_target.shape[0], steps-1))
                source_hist = np.empty((temp_source.shape[0], steps-1))

                # for each line
                for j in xrange(temp_target.shape[0]):
                    bins = np.linspace(feature_min[j], feature_max[j], steps)
                    target_hist[j], _ = np.histogram(temp_target[j], bins)
                    source_hist[j], _ = np.histogram(temp_source[j], bins)

                temp_target_changed = np.empty(temp_target.shape)
                for j in xrange(temp_target.shape[0]):
                    xInterval = self.pdf_transfer1D(target_hist[j], source_hist[j])
                    scale = (len(xInterval) - 1) / (feature_max[j] - feature_min[j])
                    temp_target_changed[j] = np.interp((temp_target[j] -
                                                        feature_min[j])*scale,
                                                       np.arange(len(xInterval)),
                                                       xInterval) / scale + feature_min[j]
                
                u, s, v = np.linalg.svd(Rotate)
                s = 1.0 / s
                S = np.zeros(Rotate.shape)
                minDim = min(Rotate.shape[0], Rotate.shape[1])
                S[:minDim, :minDim] = np.diag(s)
                r_inv = ( (v.T).dot(S.T) ).dot(u.T)
                target_features = r_inv.dot(temp_target_changed - temp_target) + target_features

            return target_features



    def pyramid_match(self):
        """
        use the Laplacian pyramid match
        """
        pass
                
        

if __name__ == '__main__':
    from skimage import io, util
    import matplotlib.pyplot as plt
    source_img = io.imread('scotland_plain.jpg')
    target_img = io.imread('scotland_house.jpg')
    source_img = util.img_as_float(source_img)
    target_img = util.img_as_float(target_img)
    source_img = source_img[:,:,:3]
    target_img = target_img[:,:,:3]
    source_shape = source_img.shape
    target_shape = target_img.shape
    source_img = source_img.reshape(-1, 3)
    target_img = target_img.reshape(-1, 3)
    fm = FeatureMatch(20)
    target_res = fm.match(target_img.transpose(), source_img.transpose())
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(target_img.reshape(target_shape))
    ax[0,1].imshow(source_img.reshape(source_shape))
    ax[1,1].imshow(target_res.T.reshape(target_shape))
    plt.show()
