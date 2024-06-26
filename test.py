# test irreveribility_estimator module
import irreversibility_estimator as ie
import numpy as np

estimator = ie.IrreversibilityEstimator
mean = 0.6
std = 1
x_forward = np.random.normal(mean,std,size=(10000,2))
x_backward = -x_forward[:,::-1]

print(estimator.fit_predict(x_forward, x_backward))
#expected KL divergence value
print( x_forward.shape[1]*2*(mean/std)**2 )