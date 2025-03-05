import matplotlib.pyplot as plt

import SimpleITK as sitk
import torch

import os
import numpy as np
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.projectors.SPECT import SPECTSystemMatrix


torch_source = torch.randn(100, 100, 100, requires_grad=True).cuda()

torch_attmap = torch.randn(100, 100, 100, requires_grad=True).cuda()

print("Source shape: ", torch_source.shape)
print("Attenuation map shape: ", torch_attmap.shape)
print("------------------------------------")

#### make metadata ####
object_meta = SPECTObjectMeta(dr = (4.7952000000000004, 4.7952000000000004, 4.7952000000000004), 
                              shape = torch_source.shape)

proj_meta = SPECTProjMeta(projection_shape= torch_source.shape[1:], 
                          dr= (4.7952000000000004, 4.7952000000000004),
                          angles= np.linspace(0, 360, 120, endpoint=False).tolist(), 
                          radii=[380] * 120)

#### make transforms ####
attenuation_transform = SPECTAttenuationTransform(torch_attmap)
psf_meta = dicom.get_psfmeta_from_scanner_params('SY-ME', energy_keV=208, intrinsic_resolution=0.38)
psf_transform = SPECTPSFTransform(psf_meta)

#### make system matrix ####
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms = [attenuation_transform, psf_transform],
    proj2proj_transforms = [],
    object_meta = object_meta,
    proj_meta = proj_meta)

### Make Projection data (Forward proj) ###
proj = system_matrix.forward(torch_source)
print("Grad ?", proj.requires_grad)
print("Grad ?", proj.grad_fn)
