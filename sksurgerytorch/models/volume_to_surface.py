"""V2SNet Model Impementation"""

import logging
import numpy as np
import torch

from sksurgerytorch.models.volume_to_surface_model import Model

LOGGER = logging.getLogger(__name__)

#pylint:disable=unused-variable, super-with-arguments, invalid-name

class Volume2SurfaceCNN:
    """Class to encapsulate network form 'Non-Rigid Volume to Surface
    Registration using a Data-Driven Biomechanical Model'.

     Thanks to
      `Micha Pfieffer <https://gitlab.com/nct_tso_public/Volume2SurfaceCNNo>`_,\
          for their network implementation.

      :param mask: If true, use maskgin
      :type mask: bool
      :param weights: Path to trained model weights (.tar file)
      :type weights: str
    """

    def __init__(self,
                 mask: bool = True,
                 weights: str = None,
                 grid_size: int = 64):

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            LOGGER.info("Using GPU")
        else:
            self.device = torch.device("cpu")
            LOGGER.info("Using CPU")

        self.mask = mask
        self.grid_size = grid_size
        self.model = Model(mask)

        if weights is not None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0)

            checkpoint = torch.load(weights)
            self.model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])

        self.model.to(self.device)
        self.model.eval()

    def predict(self,
                preoperative: np.ndarray,
                intraoperative: np.ndarray) -> np.ndarray:
        """Predict the displacement field between model and surface.

        :param preoperative:
        :type preoperative: np.ndarray
        :param intraoperative: [description]
        :type intraoperative: np.ndarray
        :raises IOError: [description]
        :return: [description]
        :rtype: np.ndarray
        """
        gs = self.grid_size
        intraoperative = np.reshape(intraoperative, (gs, gs, gs, 1))
        intraoperative = np.transpose(intraoperative, (3, 0, 1, 2))

        preoperative = np.reshape(preoperative, (gs, gs, gs, 1))
        preoperative = np.transpose(preoperative, (3, 0, 1, 2))

        preoperative = torch.FloatTensor(preoperative).to(self.device)
        intraoperative = torch.FloatTensor(intraoperative).to(self.device)

        preoperative = preoperative.unsqueeze(0)
        intraoperative = intraoperative.unsqueeze(0)

        mask = (preoperative < 0)

        # If no values in the SDF are lower than 0 then this is not a valid
        # mesh.
        if not mask.any():
            raise IOError(
                "Sample contains no internal points (no valid signed distance\
                     function?)")

        out64, out32, out16, out8 = self.model(preoperative, intraoperative)
        estimated_displacmement = (out64).squeeze()
        mask64 = (preoperative <= 0).float()

        meanDisplacement = torch.sum(
            torch.norm(
                out64 * mask64,
                dim=1)) / torch.sum(mask64)
        maxDisplacement = torch.max(torch.norm(out64 * mask64, dim=1))

        return estimated_displacmement
