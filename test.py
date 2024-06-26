# test irreveribility_estimator module
import irreversibility_estimator as ie
import numpy as np

estimator = ie.IrreversibilityEstimator()
x_forward = np.random.normal(0.6,1,size=(1000,2))
x_backward = -x_forward[:,::-1]

print(estimator.fit_predict(x_forward, x_backward))