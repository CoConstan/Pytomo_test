import matplotlib.pyplot as plt

import SimpleITK as sitk
import torch

import os
import numpy as np
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.utils import print_collimator_parameters
from torch.nn import functional as F


def save_reconstruction_mdh(reconstructed_object, filename="reconstructed_object", origin=(0,0,0), spacing=(1,1,1)):
    # Convert the PyTorch tensor to a NumPy array
    np_image = reconstructed_object.cpu().numpy()
    
    # Convert the NumPy array to a SimpleITK image
    itk_image = sitk.GetImageFromArray(np_image)
    itk_image.SetOrigin(origin)
    itk_image.SetSpacing(spacing)


    # Save the image in MDH format
    sitk.WriteImage(itk_image, '{}.mhd'.format(filename))
    
    # Save the raw data
    raw_filename = '{}.raw'.format(filename)
    with open(raw_filename, 'wb') as raw_file:
        raw_file.write(np_image.tobytes())

##### Open the source data #####
itk_source = sitk.ReadImage('Test_case/source_padded.mhd')
np_source = sitk.GetArrayFromImage(itk_source)
noised_source = np.random.poisson(np_source) ## Add noise to the source
torch_source = torch.from_numpy(noised_source).to(dtype=torch.float32).cuda()


##### Open the attenuation map #####
itk_attmap = sitk.ReadImage('Test_case/attmap_padded.mhd')
np_attmap = sitk.GetArrayFromImage(itk_attmap)
torch_attmap = torch.from_numpy(np_attmap).to(dtype=torch.float32).cuda()

print("Source shape: ", torch_source.shape)
print("Attenuation map shape: ", torch_attmap.shape)
print("------------------------------------")

#### make metadata ####
object_meta = SPECTObjectMeta(dr = (4.7952000000000004, 4.7952000000000004, 4.7952000000000004), 
                              shape = noised_source.shape)

#object_meta.pad_size = 17
#object_meta.padded_shape = (86,115, 104)
proj_meta = SPECTProjMeta(projection_shape= noised_source.shape[1:], 
                          dr= (4.7952000000000004, 4.7952000000000004),
                          angles= np.linspace(0, 360, 120, endpoint=False).tolist(), 
                          radii=[380] * 120)

#proj_meta.pad_size = 17

print("Object meta: ", object_meta)
print("Projection meta: ", proj_meta)

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

### Make Projection data ###
proj = system_matrix.forward(torch_source)
print("Projection shape: ", proj.shape)


#### make likelihood ####
likelihood = PoissonLogLikelihood(system_matrix, projections=proj.cuda())
reconstruction_algorithm = OSEM(likelihood)
reconstructed_object = reconstruction_algorithm(n_iters=4, n_subsets=8)

save_reconstruction_mdh(reconstructed_object, filename="Test_case/reconstructed_object_att_psf", origin=itk_attmap.GetOrigin(), spacing=object_meta.dr)