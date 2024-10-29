import SimpleITK as sitk
import numpy as np

          
ref_image = sitk.ReadImage('Test_case/attmap_padded.mhd')
origin = ref_image.GetOrigin()
spacing = ref_image.GetSpacing()
image_size = ref_image.GetSize()

sphere_radius = 10  # Radius of the spheres

# Create an empty image
image = sitk.Image(image_size, sitk.sitkUInt8)

# Define the center of the first and second spheres
sphere1_center = [66, 44, 43]
sphere2_center = [35, 70, 60]

# Function to add a sphere to an image
def add_sphere(image, center, radius):
    for x in range(image.GetSize()[0]):
        for y in range(image.GetSize()[1]):
            for z in range(image.GetSize()[2]):
                if np.sum((np.array([x, y, z]) - center)**2) <= radius**2:
                    image.SetPixel([x, y, z], 100)  # Set voxel inside the sphere to 1
    return image

# Add the two spheres to the image
image = add_sphere(image, sphere1_center, sphere_radius/3)
image = add_sphere(image, sphere2_center, sphere_radius)

# Set origin and center
image.SetOrigin(origin)
image.SetSpacing(spacing)

# Save the image in MetaImage (.mhd) format
sitk.WriteImage(image, "Test_case/source_padded.mhd")