#######################################################
## Spectral image analysis program
#######################################################
# 1. image classification
# 2. edge detection
# 3. image crop & save
# 4. master frame
# 5. preprocessing
# 6. cosmic ray removal
# 7. geometry
# 8. extraction
# 9. wavelength calibration
#######################################################


import os
from astropy.io import fits
import specfunc_ver6 as spf
import numpy as np


# parameter
# base_path = 'C:/obsdata'    # file path
base_path = "C:/Users/joonh/MDFTS"
obsdate = '20240828'
dW = 10                     # image crop
shift = 1500
sigma = 2                   # gaussian smoothing
l_thrsh = 30                # edge detection
r_thrsh = 50    
sigclip = 10                # cosmic ray removal
sigfrac = 5
gain = 0.454
readnoise = 9.4

'''
↓ The larger the absolute value, the more the reference value falls 
into the inner part of the Gaussian smoothing graph.
'''
l_upper_shift = 55          # l1_y 
l_lower_shift = -80         # l2_y
r_upper_shift = 100         # r1_y
r_lower_shift = -100        # r2_y

# 55, -21, 35, -25
# 55, -80, 100, -100

extract_n = 5              # extraction y-coordinate number
extract_dW = 3
extract_fwhm_multiple_bg_inner = 4
extract_fwhm_multiple_bg_outer = 6
extract_fwhm_multiple_spec_radius = 3
#######################################################################################
CheckingOnOff = input('Do you want to check each course?(Y/N): ')

# 1. image classification
parent_folder, new_folder_path = spf.cr8fld(base_path, obsdate) # create folder(spec20240627)
arc_folder, flat_folder = spf.classify_files_initial(parent_folder, new_folder_path) # image classification(flat.list, piser.list, ...)
print('Image Classification is complete.')
print(f'arc_folder : {arc_folder}') 
print(f'flat_folder : {flat_folder}')
print('=========================process01=========================')
if CheckingOnOff == 'Y':
    check01 = input('Checking Process 01(Please enter anything): ')
else:
    pass

# 2. edge detection
orig_height, orig_width = spf.original_img_size(new_folder_path)
l_gau, r_gau = spf.edge1_smoothing(new_folder_path, orig_width = orig_width, dW = dW, shift = shift, sigma = sigma, GV = False) # detect edges from flat-001
l1_y, l2_y, r1_y, r2_y = spf.edge2_detection(l_gau, r_gau, l_thrsh = l_thrsh, r_thrsh = r_thrsh, 
                                             l_upper_shift = l_upper_shift, l_lower_shift = l_lower_shift, 
                                             r_upper_shift = r_upper_shift, r_lower_shift = r_lower_shift, EV = False)
print(f'{l1_y}, {l2_y}, {r1_y}, {r2_y}')

y_max, y_min = spf.edge3_minmax(l1_y, l2_y, r1_y, r2_y, dW = dW, shift = shift, width = orig_width)
print('Extraction of the y-coordinate of the edge is completed.')
print('=========================process02=========================')
if CheckingOnOff == 'Y':
    check02 = input('Checking Process 02(Please enter anything): ')
else:
    pass

# 3. image crop & save
edge_min = int(np.ceil(y_min))
edge_max = int(np.trunc(y_max))

for img_type in ['bias', 'dark_300s', 'dark_1800s', 'piser', 'tcrb', 'arc', 'flat']:
    crop_data = spf.crop_img(img_type, new_folder_path, edge_min, edge_max)
    spf.save_img_tuple(crop_data, new_folder_path, 'crop')
    print(f'Saved : {img_type}')
print('All cropped images are saved')

spf.classify_files_by_process(new_folder_path, 'crop')
print('=========================process03=========================')
if CheckingOnOff == 'Y':
    check03 = input('Checking Process 03(Please enter anything): ')
else:
    pass

# 4. master frame
mbias = spf.create_master_bias(new_folder_path)
spf.save_mframe(mbias, new_folder_path, 'bias')
print('Saved : master bias')

mdark_300s = spf.create_master_dark(mbias, new_folder_path, 300)
spf.save_mframe(mdark_300s, new_folder_path, 'dark_300s')
print('Saved : master dark (300s)')
mdark_1800s = spf.create_master_dark(mbias, new_folder_path, 1800)
spf.save_mframe(mdark_1800s, new_folder_path, 'dark_1800s')
print('Saved : master dark (1800s)')

mflat = spf.create_master_flat(mbias, new_folder_path)
spf.save_mframe(mflat, new_folder_path, 'flat')
print('Saved : master flat')
print('=========================process04=========================')
if CheckingOnOff == 'Y':
    check04 = input('Checking Process 04(Please enter anything): ')
else:
    pass

# 5. preprocessing
for img_type in ['piser', 'tcrb', 'arc']:
    preprocess_data = spf.preprocess_img(mbias, mflat, img_type, new_folder_path)
    spf.save_img_tuple(preprocess_data, new_folder_path, 'prepro')
    print(f'Saved : {img_type}')
print('All preprocessed images are saved')

spf.classify_files_by_process(new_folder_path, 'prepro')
print('=========================process05=========================')
if CheckingOnOff == 'Y':
    check05 = input('Checking Process 05(Please enter anything): ')
else:
    pass

# 6. cosmic ray removal
for img_type in ['piser', 'tcrb', 'arc']:
    crr_data = spf.cr_removal(img_type, new_folder_path, sigclip = sigclip, sigfrac = sigfrac, gain = gain, readnoise = readnoise)
    spf.save_img_tuple(crr_data, new_folder_path, 'crr')
    print(f'Saved : {img_type}')
print('All images with cosmic ray removed are saved')

spf.classify_files_by_process(new_folder_path, 'crr')
print('=========================process06=========================')
if CheckingOnOff == 'Y':
    check06 = input('Checking Process 06(Please enter anything): ')
else:
    pass

# 7. geometry
rotation_angle1, rotation_angle2, total_angle = spf.geometry_twice(new_folder_path)
print(f'The total rotation angle is {rotation_angle1} + {rotation_angle2} = {total_angle}.')

for img_type in ['piser', 'tcrb', 'arc']:
    geo_data = spf.geometry_img(img_type, new_folder_path, total_angle)
    spf.save_img_tuple(geo_data, new_folder_path, 'geo')
    print(f'Saved : {img_type}')
print('All rotated images are saved')

spf.classify_files_by_process(new_folder_path, 'geo')
print('=========================process07=========================')
if CheckingOnOff == 'Y':
    check07 = input('Checking Process 07(Please enter anything): ')
else:
    pass

# 8. extraction
for img_type in ['piser', 'tcrb']:
# for img_type in ['tcrb']:
    ## Star Extraction 
    median_y_data = spf.extract_spectrum_y_coordi(img_type, new_folder_path, extract_n, extract_dW)
    fwhm_data = spf.compute_fwhm(img_type, new_folder_path, median_y_data, sigma=2, n_segments=10)
    background_values = spf.compute_background(img_type, new_folder_path, median_y_data, fwhm_data, fwhm_inner=5, fwhm_outer=15, plotCheck=False)
    spectrum_values = spf.compute_spectrum(img_type, new_folder_path, median_y_data, fwhm_data, background_values, fwhm_spectrum=4, plotCheck=False)
    spectrum_values = spf.trim_spectrum_data(spectrum_values, delete_data=10, plotCheck=False, compare=False)
    
    ## Arc Extraction
    extracted_spectrum_list = spf.extract_arc_spectrum(img_type, new_folder_path, median_y_data, fwhm_data, fwhm_arc_spectrum=4, plotCheck=False)
    extracted_spectrum_list = spf.trim_arc_data(extracted_spectrum_list, delete_data=10, plotCheck=False, compare=False)
    
    ## Save data to FITS
    spf.save_spectrum_data_to_FITS(img_type, new_folder_path, spectrum_values, extracted_spectrum_list)
print('Extractions are completed')

print('=========================process08=========================')
if CheckingOnOff == 'Y':
    check08 = input('Checking Process 08(Please enter anything): ')
else:
    pass


# 9. wavelength calibration
for img_type in ['piser', 'tcrb']:
    spf.wavelength_equation(img_type, new_folder_path, sigma=2)
    print(f'Done : {img_type}')
print('Wavelength calibrations are completed')

print('=========================process09=========================')
if CheckingOnOff == 'Y':
    check08 = input('Checking Process 09(Please enter anything): ')
else:
    pass