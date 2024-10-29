import numpy as np
import SimpleITK as sitk

def get_profile_from_sphere(image, sphere_center, sphere_radius):
    profile = []
    for i in range((sphere_radius+4)*2):
        x = sphere_center[0] - sphere_radius - 2 + i
        y = sphere_center[1]
        z = sphere_center[2]
        profile.append(image.GetPixel([x, y, z]))
    return profile

def plot_profile(profile):
    import matplotlib.pyplot as plt
    plt.plot(profile)
    plt.show()

im_path = 'Test_case/reconstructed_object_att_psf.mhd'
sphere1_center = [66, 44, 43]
sphere2_center = [35, 70, 60]

##### Open the source data #####
itk_image = sitk.ReadImage(im_path)
np_image= sitk.GetArrayFromImage(itk_image)

profile1 = get_profile_from_sphere(itk_image, sphere1_center, 10)
profile2 = get_profile_from_sphere(itk_image, sphere2_center, 10)

#print(len(profile1))
#print(len(profile2))

plot_profile(profile1)
plot_profile(profile2)

