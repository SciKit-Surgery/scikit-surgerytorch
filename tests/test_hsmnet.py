import os
import pytest
import cv2
import numpy as np
from sksurgerytorch.models import high_res_stereo


def test_hsmnet_no_weights():
    """ This won't give a good results, but we can check that the network
    at least runs ok. """
    max_disp = 255
    entropy_threshold = -1
    scale_factor = 1.0
    level = 1

    HSMNet = high_res_stereo.HSMNet(max_disp=max_disp,
                                    entropy_threshold=entropy_threshold,
                                    level=level,
                                    scale_factor=scale_factor,
                                    weights=None)

    left = cv2.imread('tests/data/hsmnet/synthetic-l.png')
    right = cv2.imread('tests/data/hsmnet/synthetic-r.png')


    disp, entropy = HSMNet.predict(left, right)

    cv2.imwrite('tests/output/synthetic_disp_no_training.png', disp)

weights = ('tests/data/weights/final-768px.tar')

@pytest.mark.skipif(not os.path.exists(weights), reason="Weights not found")
def test_hsmnet_pretrained_weights():

    max_disp = 255
    entropy_threshold = -1
    scale_factor = 1.0
    level = 1

    HSMNet = high_res_stereo.HSMNet(max_disp=max_disp,
                                    entropy_threshold=entropy_threshold,
                                    level=level,
                                    scale_factor=scale_factor,
                                    weights=weights)

    left = cv2.imread('tests/data/hsmnet/synthetic-l.png')
    right = cv2.imread('tests/data/hsmnet/synthetic-r.png')


    disp, entropy = HSMNet.predict(left, right)
    print(disp.dtype)
    expected_disp = cv2.imread(
        'tests/data/hsmnet/synthetic-expected-disp.png', -1).astype(np.float32)

    np.testing.assert_array_almost_equal(disp, expected_disp, decimal=0)
    #cv2.imwrite('tests/output/hsmnet_disp.png', disp)

