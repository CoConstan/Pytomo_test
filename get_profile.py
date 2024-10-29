import SimpleITK as sitk
import matplotlib.pyplot as plt

def get_profile_from_sphere(image, sphere_center, sphere_radius):
    profile = []
    for i in range((sphere_radius+4)*2):
        x = sphere_center[0] - sphere_radius - 2 + i
        y = sphere_center[1]
        z = sphere_center[2]
        profile.append(image.GetPixel([x, y, z]))
    return profile

def plot_profile(profile, path = "Fig/", title='Profile', xlabel='Index', ylabel='Value'):
    
    plt.plot(profile)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path + title + '.png')
    #plt.show()
    plt.close()

sphere1_center = [66, 44, 43]
sphere2_center = [35, 70, 60]

source_path = 'Test_case_2/source_padded.mhd'
reconstructed_path_1 = 'Test_case_2/reconstructed_object_no_att_no_psf.mhd'
reconstructed_path_2 = 'Test_case_2/reconstructed_object_att_no_psf.mhd'
reconstructed_path_3 = 'Test_case_2/reconstructed_object_att_psf.mhd'

shpere1_profile = []
shpere2_profile = []

##### Profile from the source #####
source_image = sitk.ReadImage(source_path)

profile1 = get_profile_from_sphere(source_image, sphere1_center, 10)
profile2 = get_profile_from_sphere(source_image, sphere2_center, 10)
plot_profile(profile1, title='Profile_S1_source')
plot_profile(profile2, title='Profile_S2_source')

shpere1_profile.append(profile1)
shpere2_profile.append(profile2)

##### Profile from the reconstructed image without att and psf #####
reconstructed_image_1 = sitk.ReadImage(reconstructed_path_1)

profile1 = get_profile_from_sphere(reconstructed_image_1, sphere1_center, 10)
profile2 = get_profile_from_sphere(reconstructed_image_1, sphere2_center, 10)
plot_profile(profile1, title='Profile_S1_no_att_no_psf')
plot_profile(profile2, title='Profile_S2_no_att_no_psf')

shpere1_profile.append(profile1)
shpere2_profile.append(profile2)

##### Profile from the reconstructed image with att and without psf #####

reconstructed_image_2 = sitk.ReadImage(reconstructed_path_2)

profile1 = get_profile_from_sphere(reconstructed_image_2, sphere1_center, 10)
profile2 = get_profile_from_sphere(reconstructed_image_2, sphere2_center, 10)
plot_profile(profile1, title='Profile_S1_att_no_psf')
plot_profile(profile2, title='Profile_S2_att_no_psf')

shpere1_profile.append(profile1)
shpere2_profile.append(profile2)

##### Profile from the reconstructed image with att and psf #####

reconstructed_image_3 = sitk.ReadImage(reconstructed_path_3)

profile1 = get_profile_from_sphere(reconstructed_image_3, sphere1_center, 10)
profile2 = get_profile_from_sphere(reconstructed_image_3, sphere2_center, 10)
plot_profile(profile1, title='Profile_S1_att_psf')
plot_profile(profile2, title='Profile_S2_att_psf')

shpere1_profile.append(profile1)
shpere2_profile.append(profile2)

##### Plot profiles for the spheres 1 and 2 #####

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotting on the first subplot
ax1.plot(shpere1_profile[0], label="Source Data", linestyle="-", linewidth=2)
ax1.plot(shpere1_profile[1], label="Reconstruction", linestyle="--")
ax1.plot(shpere1_profile[2], label="Reconstruction att", linestyle="--")
ax1.plot(shpere1_profile[3], label="Reconstruction att + psf", linestyle="--")

# Add more plt.plot() for additional reconstruction data on the first subplot
ax1.set_xlabel("Position")
ax1.set_ylabel("Intensity")
ax1.set_title("Profiles from Sphere 1")
ax1.legend()
ax1.grid(True)

# Plotting on the second subplot
# Plotting on the first subplot
ax2.plot(shpere2_profile[0], label="Source Data", linestyle="-", linewidth=2)
ax2.plot(shpere2_profile[1], label="Reconstruction", linestyle="--")
ax2.plot(shpere2_profile[2], label="Reconstruction att", linestyle="--")
ax2.plot(shpere2_profile[3], label="Reconstruction att + psf", linestyle="--")

# Add more plt.plot() for additional reconstruction data on the second subplot
ax2.set_xlabel("Position")
ax2.set_ylabel("Intensity")
ax2.set_title("Profiles from Sphere 2")
ax2.legend()
ax2.grid(True)
# Adjust layout to prevent overlap
plt.tight_layout()

plt.savefig('Fig/Profiles.png')
# Show plot
#plt.show()

