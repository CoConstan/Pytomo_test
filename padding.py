import SimpleITK as sitk

def pad_itk_image(input_image, pad_lower, pad_upper, pad_value=0):
    """
    Pads an ITK image with the specified padding values.

    Parameters:
    - input_image: The input ITK image.
    - pad_lower: A list or tuple specifying the amount of padding to add to the lower end of each dimension.
    - pad_upper: A list or tuple specifying the amount of padding to add to the upper end of each dimension.
    - pad_value: The value to use for the padding (default is 0).

    Returns:
    - The padded ITK image.
    """
    # Create a ConstantPadImageFilter
    pad_filter = sitk.ConstantPadImageFilter()
    
    # Set the lower and upper padding values
    pad_filter.SetPadLowerBound(pad_lower)
    pad_filter.SetPadUpperBound(pad_upper)
    
    # Set the padding value
    pad_filter.SetConstant(pad_value)
    
    # Apply the padding filter
    padded_image = pad_filter.Execute(input_image)
    
    return padded_image

# Example usage:
# Read the input image
input_image = sitk.ReadImage('Test_case/attmap.mhd')

# Define the padding values for each dimension
pad_lower = [0, 12, 22]  # Padding to add to the lower end of each dimension
pad_upper = [0, 11, 22]  # Padding to add to the upper end of each dimension

# Pad the image
padded_image = pad_itk_image(input_image, pad_lower, pad_upper, pad_value=0)

np_input_image = sitk.GetArrayFromImage(input_image)
np_padded_image = sitk.GetArrayFromImage(padded_image)
print("Input image shape: ", np_input_image.shape)
print("Padded image shape: ", np_padded_image.shape)

# Save the padded image
sitk.WriteImage(padded_image, 'Test_case/attmap_padded.mhd')