"""HSMNet Tests."""
import os
import pytest
import numpy as np
import torch
from sksurgerytorch.models.volume_to_surface import Volume2SurfaceCNN

#pylint:disable=not-callable,invalid-name
def test_v2snet_no_weights():
    """ This won't give a good results, but we can check that the network
    at least runs ok. """

    grid_size = 64
    num_elements = grid_size*grid_size*grid_size
    V2SNet = Volume2SurfaceCNN(grid_size=grid_size)

    preop_signed = np.random.random(num_elements) - 0.5
    intraop_unsigned = np.random.random(num_elements)

    V2SNet.predict(preop_signed, intraop_unsigned)

weights = ('tests/data/weights/v2s_80k_micha.tar')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not found")
@pytest.mark.skipif(not os.path.exists(weights), reason="Weights not found")
def test_v2snet_with_weights():

    """ Test with some (non-trained) weights."""
    grid_size = 64
    V2SNet = Volume2SurfaceCNN(grid_size=grid_size, weights=weights)

    preop_signed = np.load('tests/data/v2snet/preop.npy')
    intraop_unsigned =np.load('tests/data/v2snet/intraop.npy')

    displacement = V2SNet.predict(preop_signed, intraop_unsigned)

    expected_displacement = \
        np.load('tests/data/v2snet/estimatedDisplacement.npy')

    assert np.allclose(displacement, expected_displacement)
