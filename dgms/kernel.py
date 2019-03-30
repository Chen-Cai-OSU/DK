import numpy as np
import time
import sklearn_tda as tda
from joblib import Parallel, delayed

from dgms.test import generate_swdgm
from helper.format import precision_format
def sw(dgms1, dgms2, parallel_flag=False, kernel_type='sw', n_directions=10, bandwidth=1.0, K=1, p = 1):
    # dgms1, dgms2 here are numpy array
    def arctan(C, p):
        return lambda x: C * np.arctan(np.power(x[1], p))
    if parallel_flag==False:
        if kernel_type=='sw':
            tda_kernel = tda.SlicedWassersteinKernel(num_directions=n_directions, bandwidth=bandwidth)
        elif kernel_type=='pss':
            tda_kernel = tda.PersistenceScaleSpaceKernel(bandwidth=bandwidth)
        elif kernel_type == 'wg':
            tda_kernel = tda.PersistenceWeightedGaussianKernel(bandwidth=bandwidth, weight=arctan(K, p))
        else:
            print ('Unknown kernel')

        diags1 = dgms1; diags2 = dgms2
        X = tda_kernel.fit(diags1)
        Y = tda_kernel.transform(diags2)
        return Y

def sw_parallel(dgms1, dgms2,  kernel_type='sw', parallel_flag=True, n_directions=10, granularity=25, **kwargs):
    # dgms1: a list of array. kwargs: contain bw;

    t1 = time.time()
    assert_sw_dgm(dgms1)
    assert_sw_dgm(dgms2)
    n1 = len(dgms1); n2 = len(dgms2)
    kernel = np.zeros((n2, n1))

    if parallel_flag:
        # parallel version
        kernel = Parallel(n_jobs=-1)(delayed(sw)(dgms1, dgms2[i:min(i+granularity, n2)], kernel_type=kernel_type,
                                                 n_directions=n_directions, bandwidth=kwargs['bw'], K=kwargs['K'],
                                                 p=kwargs['p']) for i in range(0, n2, granularity))
        kernel = (np.vstack(kernel))
    else: # used as verification
        for i in range(n2):
            kernel[i] = sw(dgms1, [dgms2[i]], kernel_type=kernel_type, n_directions=n_directions, bandwidth=kwargs['bw'])

    t = precision_format(time.time()-t1, 1)
    print('Finish computing %s kernel of shape %s. Takes %s'%(kernel_type, kernel.shape, t))
    return (kernel/float(np.max(kernel)), t)

def sw_parallel_test():
    dgms1 = generate_swdgm(1000)
    dummy_kwargs = {'K':1, 'p':1}
    serial_kernel = sw_parallel(dgms1, dgms1, bw=1, parallel_flag=False, **dummy_kwargs)[0]
    parallel_kernel = sw_parallel(dgms1, dgms1, bw=1, parallel_flag=True, **dummy_kwargs)[0]
    diff = serial_kernel - parallel_kernel
    assert np.max(diff) < 1e-5


def assert_sw_dgm(dgms):
    # check sw_dgm is a list array
    # assert_sw_dgm(generate_swdgm(10))
    assert type(dgms)==list
    for dgm in dgms:
        try:
            assert np.shape(dgm)[1]==2
        except AssertionError:
            print('Not the right format for sw. Should be a list of array')
