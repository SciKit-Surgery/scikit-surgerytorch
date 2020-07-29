from sksurgerytorch.models import high_res_stereo
import cv2

def test_hsmnet():

    max_disp = 255
    entropy_threshold = -1
    scale_factor = 1.0
    level = 1
    weights = ('../high-res-stereo/weights/final-768px.tar')

    HSMNet = high_res_stereo.HSMNet(max_disp=max_disp,
                                    entropy_threshold=entropy_threshold,
                                    level=level,
                                    scale_factor=scale_factor,
                                    weights=weights)

    left = cv2.imread('tests/data/hsmnet/synthetic-l.png')
    right = cv2.imread('tests/data/hsmnet/synthetic-r.png')


    disp, entropy = HSMNet.predict(left, right)

    cv2.imwrite('tests/output/hsmnet_disp.png', disp)