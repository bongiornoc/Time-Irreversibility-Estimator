# test irreveribility_estimator module
import irreversibility_estimator as ie
import numpy as np

estimator = ie.IrreversibilityEstimator(verbose=True)
mean = 0.6
std = 1
x_forward = np.random.normal(mean,std,size=(10000,2))
x_backward = -x_forward[:,::-1]

print(estimator.fit_predict(x_forward, x_backward))
#expected KL divergence value
print( x_forward.shape[1]*2*(mean/std)**2 )

print(estimator.fit_predict(x_forward, return_log_diffs=True))

groups = np.random.randint(0,6,10000)
print(estimator.fit_predict(x_forward, x_backward, groups=groups))


x_forward = np.random.normal(mean,std,size=(1000))
print(estimator.fit_predict(x_forward))