# -*- coding: utf-8 -*-

"""
Module to implement Hierarchical Deep Stereo Matching on High Resolution Images
network.
"""

import os
import sys
import logging
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms

from high_res_stereo_model import HSMNet_model

LOGGER = logging.getLogger(__name__)

class HSMNet:
    """Class to encapsulate network form 'Hierarchical Deep Stereo Matching on
     High Resolution Images'.

     Thanks to
      `Gengshang Yang <https://github.com/gengshan-y/high-res-stereo>`_, for
      their network implementation.

      :param max_disp: Maximum number of disparity levels
      :param entropy_threshold: Pixels with entropy above this value will be
      ignored in the disparity map. Disabled if set to -1.
      :param level: Set to 1, 2 or 3 to trade off quality of depth estimation
       against runtime. 1 = best depth estimation, longer runtime,
        3 = worst depth estimation, fastest runtime.
      :param scale_factor: Images can be resized before passing to the network,
       for perfomance impromvents. This sets the scale factor. 
       :param weights: Path to trained model weights (.tar file)
    """

    def __init__(self, max_disp:int=255, entropy_threshold:float=-1, level:int=1, scale_factor:float=0.5, weights=None):

        self.max_disp = maxdisp
        self.test_res = testres
        self.entropy_threshold = entropy_threshold
        self.level = level

        self.model = HSMNet_model(maxdisp, entropy_threshold, level)
        self.model.eval()

        self.pred_disp = None
        self.entropy = None

        if weights:
            LOGGER.info("Loading weights from %s", weights)
            pretrained_dict = torch.load(weights)
            pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
            model.load_state_dict(pretrained_dict['state_dict'],strict=False)
            LOGGER.info("Loaded weights")

        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))



    def predict(self,
                left_image: np.ndarray,
                right_image: np.ndarray)-> np.ndarray:
        """Predict disparity from a pair of stereo images.

        :param left_image: Left stereo image, 3 channel RGB
        :type left_image: np.ndarray
        :param right_image: Right stero image, 3 channel RGB
        :type right_image: np.ndarray
        :return: Predicted disparity, grayscale
        :rtype: np.ndarray
        """
        t_list = [
        toTensorLegacy(),
        transforms.Normalize(**__imagenet_stats),
        ]

        processed = transforms.Compose(t_list)

        #TODO: Move this to init or model bit?
        disp_scaled = int(self.max_disp * self.test_res // 64 * 64)
        self.model.module.maxdisp = disp_scaled 
        if self.model.module.maxdisp == 64: self.model.module.maxdisp=128
        self.model.module.disp_reg8 =  disparityregression(self.model.module.maxdisp,16).cuda()
        self.model.module.disp_reg16 = disparityregression(self.model.module.maxdisp,16).cuda()
        self.model.module.disp_reg32 = disparityregression(self.model.module.maxdisp,32).cuda()
        self.model.module.disp_reg64 = disparityregression(self.model.module.maxdisp,64).cuda()

        LOGGER.info("Model.module.maxdisp %s", self.model.module.maxdisp)

        # resize
        imgL_o = cv2.resize(left_image,None,fx=self.test_res,fy=self.test_res,interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(right_image,None,fx=self.test_res,fy=self.test_res,interpolation=cv2.INTER_CUBIC)
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()

        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h-imgL.shape[2]
        left_pad = max_w-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            self.pred_disp, self.entropy = model(imgL,imgR)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )

        self.pred_disp = torch.squeeze(self.pred_disp).data.cpu().numpy()

        top_pad   = max_h-imgL_o.shape[0]
        left_pad  = max_w-imgL_o.shape[1]
        self.entropy = self.entropy[top_pad:,:self.pred_disp.shape[1]-left_pad].cpu().numpy()
        self.pred_disp = self.pred_disp[top_pad:,:self.pred_disp.shape[1]-left_pad]

        # save predictions
        idxname = test_left_img[inx].split('/')[-2]
        if not os.path.exists('%s/%s'%(args.outdir,idxname)):
            os.makedirs('%s/%s'%(args.outdir,idxname))
        idxname = '%s/disp0HSM'%(idxname)

        # resize to highres
        self.pred_disp = cv2.resize(self.pred_disp/args.testres,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(self.pred_disp == np.inf,self.pred_disp!=self.pred_disp)
        self.pred_disp[invalid] = np.inf

        torch.cuda.empty_cache()

        return self.pred_disp, self.entropy
