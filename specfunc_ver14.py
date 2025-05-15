#######################################################
## function definition
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
import sys
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import astroscrappy
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.optimize import least_squares
import glob


#######################################################################################


# 1. image classification
# 1-1) Create a new folder to save file lists and created images.
def cr8fld(base_path, obsdate):
    # Base folder path
    parent_folder = os.path.join(base_path, obsdate)
    
    # Check if the parent folder exists, create it if not
    if not os.path.exists(parent_folder):
        print(f'Error: Parent folder "{parent_folder}" does not exist.')
        sys.exit(1)
    else:
        print(f'"{parent_folder}" exists.')

    # Create the new folder name
    new_folder_name = f"spec{obsdate}"
    new_folder_path = os.path.join(parent_folder, new_folder_name)
    
    # If the folder already exists, append a number
    counter = 1
    while os.path.exists(new_folder_path):
        new_folder_name = f"spec{obsdate}_{counter}"
        new_folder_path = os.path.join(parent_folder, new_folder_name)
        counter += 1
        
    # Create the new folder
    os.makedirs(new_folder_path)
    print(f'Created the folder: {new_folder_path}')

    return parent_folder, new_folder_path


# 1-2) Create text files that categorize files by image type.
def classify_files_initial(parent_folder, new_folder_path):
    # File categories
    bias_files = []
    dark_300s_files = []
    dark_1800s_files = []
    piser_files = []
    tcrb_files = []
    arc_files = []
    flat_files = []

    # Iterate over files in the parent folder
    for file_name in os.listdir(parent_folder):
        if os.path.isfile(os.path.join(parent_folder, file_name)):
            # Convert the filename to lowercase for case-insensitive comparison
            uniform_file_name = file_name.lower().replace(' ', '')
            
            # Bias, Dark, PiSer, TCrB
            if 'bias' in uniform_file_name:
                bias_files.append(file_name)
            elif 'dark' and '300s' in uniform_file_name:
                dark_300s_files.append(file_name)
            elif 'dark' and '1200s' in uniform_file_name:
                dark_1800s_files.append(file_name)
            elif 'piser' in uniform_file_name:
                piser_files.append(file_name)
            elif 'tcrb' in uniform_file_name:
                tcrb_files.append(file_name)

    # Sort the files alphabetically
    bias_files.sort()
    dark_300s_files.sort()
    dark_1800s_files.sort()
    piser_files.sort()
    tcrb_files.sort()

    # Write the sorted file names to .list files
    write_to_list_file(parent_folder, new_folder_path, 'bias.list', bias_files)
    write_to_list_file(parent_folder, new_folder_path, 'dark_300s.list', dark_300s_files)
    write_to_list_file(parent_folder, new_folder_path, 'dark_1800s.list', dark_1800s_files)
    write_to_list_file(parent_folder, new_folder_path, 'piser.list', piser_files)
    write_to_list_file(parent_folder, new_folder_path, 'tcrb.list', tcrb_files)

    # Now handle arc and flat files with directory traversal
    arc_folder, flat_folder = find_arc_and_flat_files(parent_folder, arc_files, flat_files)

    # Write arc and flat files to .list files
    write_to_list_file(arc_folder, new_folder_path, 'arc.list', arc_files)
    write_to_list_file(flat_folder, new_folder_path, 'flat.list', flat_files)

    print("Files have been categorized and saved to .list files.")
    
    return arc_folder, flat_folder
    
def find_arc_and_flat_files(start_folder, arc_files, flat_files):
    arc_folder = None
    flat_folder = None

    # Get the parent folder and list of sibling folders
    parent_folder = os.path.dirname(start_folder)
    sibling_folders = sorted(os.listdir(parent_folder), reverse = True)  # Get sibling folders and sort them by name

    # Find the current folder's index
    current_folder_index = sibling_folders.index(os.path.basename(start_folder))

    # Search current folder and all "sibling" folders upwards
    for folder_name in sibling_folders[current_folder_index:]:
        sibling_folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(sibling_folder_path):
            # Search for arc and flat files in the current sibling folder
            arc_folder = search_for_files(sibling_folder_path, 'arc', arc_files)
            flat_folder = search_for_files(sibling_folder_path, 'flat', flat_files)

            # If both arc and flat files are found, exit the loop
            if arc_folder and flat_folder:
                break

    # Sort the files alphabetically if found
    arc_files.sort()
    flat_files.sort()

    # Debug print: Check what files were found
    print("Arc files found:", arc_files)
    print("Flat files found:", flat_files)

    # Return the folders where arc and flat were found
    return arc_folder, flat_folder

def search_for_files(folder_path, prefix, file_list):
    found_files = False
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            # Convert the filename to lowercase for case-insensitive comparison
            lower_file_name = file_name.lower()
            if lower_file_name.startswith(prefix):
                file_list.append(file_name)
                found_files = True
    if found_files:
        print(f"Found '{prefix}' in folder: {os.path.basename(folder_path)}")  # Debug message
        return folder_path
    return None

def write_to_list_file(parent_folder, new_folder_path, file_name, files):
    # Write the sorted file names to the .list file in the new folder path
    list_file_path = os.path.join(new_folder_path, file_name)
    if files:  # If there are files in the list
        with open(list_file_path, 'w') as f:
            for file in files:
                full_file_path = os.path.join(parent_folder, file).replace("\\", "/")
                f.write(full_file_path + '\n')
        print(f"Written {len(files)} files to {list_file_path}")
    else:
        print(f"No files found for {file_name}, skipping write.")
        
        
# 2. edge detection
# 2-1) edge1_smoothing : 
def edge1_smoothing(new_folder_path, orig_width, dW, shift, sigma, GV = False):
    # Select path for first image among flat images
    flat_list_path = os.path.join(new_folder_path, 'flat.list')
    flat_image_path = path_of_the_first_flat(flat_list_path)
    hdu = fits.open(flat_image_path)[0]
    dat = hdu.data
    
    # Extract data of dW interval from both sides of image
    l_dat = dat[:, shift : (shift + dW + 1)]
    r_dat = dat[:, (orig_width - dW - 1) : orig_width]
     
    # Convert data into one line with median
    l_med = np.median(l_dat, axis = 1)
    r_med = np.median(r_dat, axis = 1)
    
    # Gaussian Filter Smoothing
    l_gau = gaussian_filter1d(l_med, sigma = sigma)
    r_gau = gaussian_filter1d(r_med, sigma = sigma)
    
    # Visualize comparison before and after Gaussian filter
    if GV == False:
        pass
    else:
        median_n_gaussian(l_med, l_gau, r_med, r_gau)
        
    return l_gau, r_gau

def original_img_size(new_folder_path):
    # Select path for first image among flat images
    flat_list_path = os.path.join(new_folder_path, 'flat.list')
    flat_image_path = path_of_the_first_flat(flat_list_path)
    with fits.open(flat_image_path) as hdu:
        dat = hdu[0].data
        height, width = dat.shape
    
    # Extract data of dW interval from both sides of image
    height, width = dat.shape
    
    return height, width

def path_of_the_first_flat(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            return os.path.join(os.path.dirname(file_path), first_line).replace("\\", "/")
    except:
        return None
    
def median_n_gaussian(l_med, l_gau, r_med, r_gau):
    # 1x2 subplot 생성
    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    
    # 첫 번째 subplot에 l_med와 l_gau를 그림
    axs[0].plot(l_med, label = 'l_med')
    axs[0].plot(l_gau, label = 'l_gau')
    axs[0].set_title('l_med vs l_gau')
    axs[0].legend()

    # 두 번째 subplot에 r_med와 r_gau를 그림
    axs[1].plot(r_med, label = 'r_med')
    axs[1].plot(r_gau, label = 'r_gau')
    axs[1].set_title('r_med vs r_gau')
    axs[1].legend()

    # 레이아웃을 조정하여 플롯 간의 간격을 맞춤
    plt.tight_layout()

    # 플롯을 화면에 출력
    plt.show()
    
# 2-2) edge2_detection : 
def edge2_detection(l_gau, r_gau, l_thrsh, r_thrsh, l_upper_shift, l_lower_shift, r_upper_shift, r_lower_shift, EV = False):
    # Extract the first point where the slope changes sharply on the left and right
    l_edges = detect_edges(l_gau, l_thrsh)
    r_edges = detect_edges(r_gau, r_thrsh)
    
    # y-coordinate of edge
    l1_y = l_edges[0] + l_upper_shift
    l2_y = l_edges[-1] + l_lower_shift
    r1_y = r_edges[0] + r_upper_shift
    r2_y = r_edges[-1] + r_lower_shift
    
    # Visualize the location of edges in Gaussian data
    if EV == False:
        pass
    else:
        plot_edges_on_GaussianData(l_gau, r_gau, l1_y, l2_y, r1_y, r2_y)
        
    return l1_y, l2_y, r1_y, r2_y
    
def detect_edges(data, threshold):    
    # First-order differential calculationsv
    diff = np.diff(data)
    
    # Find all locations where the slope increase/decrease is greater than a certain value    
    edge_indices = np.where(np.abs(diff) > threshold)[0]
    
    return edge_indices.tolist()

def plot_edges_on_GaussianData(l_gau, r_gau, l1_y, l2_y, r1_y, r2_y):
    # 1x2 서브플롯을 생성하여 옆으로 배치
    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    
    # 첫 번째 subplot에 l_gau와 두 상수값에 해당하는 세로 점선을 그림
    axs[0].plot(l_gau, label='l_gau')
    axs[0].axvline(x = l1_y, color = 'r', linestyle = '--', label = 'l1_y')
    axs[0].axvline(x = l2_y, color = 'b', linestyle = '--', label = 'l2_y')
    axs[0].set_title('l_gau with l1_y and l2_y')
    axs[0].legend()

    # 두 번째 subplot에 r_gau와 두 상수값에 해당하는 세로 점선을 그림
    axs[1].plot(r_gau, label = 'r_gau')
    axs[1].axvline(x = r1_y, color = 'r', linestyle = '--', label = 'r1_y')
    axs[1].axvline(x = r2_y, color = 'b', linestyle = '--', label = 'r2_y')
    axs[1].set_title('r_gau with r1_y and r2_y')
    axs[1].legend()

    # 레이아웃을 조정하여 플롯 간의 간격을 맞춤
    plt.tight_layout()

    # 플롯을 화면에 출력
    plt.show()
    
# 2-3) edge3_minmax
def edge3_minmax(l1_y, l2_y, r1_y, r2_y, dW, shift, width):
    # x-coordinate of edge
    l_x = shift + dW/2
    r_x = width - dW/2
    
    # Calculate the equation of a straight line
    slope1, intercept1 = line_calc(l_x, l1_y, r_x, r1_y)
    slope2, intercept2 = line_calc(l_x, l2_y, r_x, r2_y)
    
    # Top and bottom values ​​of edge y coordinate
    y_max, y_min = edge_y_minmax(slope1, intercept1, slope2, intercept2, width)
    
    return y_max, y_min

def line_calc(l_x, l_y, r_x, r_y):
    slope = (r_y - l_y) / (r_x - l_x)
    intercept = l_y - slope * l_x
    return slope, intercept

def edge_y_minmax(slope1, intercept1, slope2, intercept2, width):
    # line1과 line2의 x좌표를 0과 width로 설정하여 y좌표 계산
    y1_start = intercept1 # y1_start = slope1 * 0 + intercept1 / x = 0
    y1_end = slope1 * width + intercept1  # x = width
    
    y2_start = intercept2 # y2_start = slope2 * 0 + intercept2  / x = 0
    y2_end = slope2 * width + intercept2  # x = width

    # y좌표 중에서 최대값과 최소값을 계산
    y_min = max([y1_start, y1_end])
    y_max = min([y2_start, y2_end])

    return y_max, y_min

# 3. image crop & save
# 3-1) crop_img
# 가로로도 자르는 함수
# def crop_img(img_type, new_folder_path, edge_min, edge_max, horizontal_crop):
    # cropped_data_list = []
    
    # list_file_path = os.path.join(new_folder_path, f'{img_type}.list')
    
    # # Open the .list file and process each file path
    # with open(list_file_path, 'r') as file:
    #     for line in file:
    #         file_path = line.strip()
    #         if not file_path:
    #             continue

    #         # Open the FITS file
    #         with fits.open(file_path) as hdul:
    #             img = hdul[0].data  # Get the image data
                
    #             # Get image dimensions
    #             height, width = img.shape

    #             # Apply vertical crop
    #             cropped_img = img[edge_min:edge_max, :]

    #             # Apply horizontal crop
    #             left_crop = horizontal_crop
    #             right_crop = width - horizontal_crop
                
    #             if left_crop < right_crop:
    #                 cropped_img = cropped_img[:, left_crop:right_crop]
    #             else:
    #                 raise ValueError("Invalid horizontal_crop: Results in zero or negative width.")

    #             # Append the file path and cropped data to the list
    #             cropped_data_list.append((file_path, cropped_img))

    # return cropped_data_list

#세로만 자르는 함수
def crop_img(img_type, new_folder_path, edge_min, edge_max):
    # img_type : bias, dark_300s, flat, piser, ...
    cropped_data_list = []
    
    list_file_path = new_folder_path + '/' + img_type + '.list'
    
    # Open the .list file and process each file path
    with open(list_file_path, 'r') as file:
        for line in file:
            # Strip whitespace and skip empty lines
            file_path = line.strip()

            # Open the FITS file
            with fits.open(file_path) as hdul:
                img = hdul[0].data  # Get the image data

                # Crop the image
                cropped_img = img[edge_min:edge_max, :]  # Crop based on input

                # Append the file path and cropped data to the list
                cropped_data_list.append((file_path, cropped_img))

    return cropped_data_list

# 3-2) save_img
def save_img_tuple(data_list, new_folder_path, process):
    # data_list : [("C:/data/obs1.fits", array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
    # ("C:/data/obs2.fits", array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])) ]
    # process : crop, geometry, ...
    for original_file_path, changed_data in data_list:
        # Extract the original file name 
        original_file_name = os.path.basename(original_file_path)
        # Construct the new file name
        new_file_name = f"{process}_{original_file_name}"
        
        # Full path for the new file
        save_path = os.path.join(new_folder_path, new_file_name)
        
        # Save the changed image as a FITS file
        hdu = fits.PrimaryHDU(changed_data)
        hdu.writeto(save_path, overwrite=True)
        
    print(f"Saved: {process}")
    
# 3-3) Sort again into cropped images
def classify_files_by_process(new_folder_path, process):
    # File categories
    bias_files = []
    dark_300s_files = []
    dark_1800s_files = []
    piser_files = []
    tcrb_files = []
    arc_files = []
    flat_files = []

    # Iterate through files in the folder
    for file_name in os.listdir(new_folder_path):
        # Check if file starts with the given process prefix
        if file_name.lower().startswith(process.lower()):
            # Convert the filename to lowercase for case-insensitive comparison
            uniform_file_name = file_name.lower().replace(' ', '')
            
            # Classify files into categories
            if 'bias' in uniform_file_name:
                bias_files.append(file_name)
            elif 'dark' and '300s' in uniform_file_name:
                dark_300s_files.append(file_name)
            elif 'dark' and '1800s' in uniform_file_name:
                dark_1800s_files.append(file_name)
            elif 'piser' in uniform_file_name:
                piser_files.append(file_name)
            elif 'tcrb' in uniform_file_name:
                tcrb_files.append(file_name)
            elif 'arc' in uniform_file_name:
                arc_files.append(file_name)
            elif 'flat' in uniform_file_name:
                flat_files.append(file_name)

    # Sort the files alphabetically
    bias_files.sort()
    dark_300s_files.sort()
    dark_1800s_files.sort()
    piser_files.sort()
    tcrb_files.sort()
    arc_files.sort()
    flat_files.sort()

    # Save sorted files to .list files
    write_to_list_file(new_folder_path, new_folder_path, f'{process}_bias.list', bias_files)
    write_to_list_file(new_folder_path, new_folder_path, f'{process}_dark_300s.list', dark_300s_files)
    write_to_list_file(new_folder_path, new_folder_path, f'{process}_dark_1800s.list', dark_1800s_files)
    write_to_list_file(new_folder_path, new_folder_path, f'{process}_piser.list', piser_files)
    write_to_list_file(new_folder_path, new_folder_path, f'{process}_tcrb.list', tcrb_files)
    write_to_list_file(new_folder_path, new_folder_path, f'{process}_arc.list', arc_files)
    write_to_list_file(new_folder_path, new_folder_path, f'{process}_flat.list', flat_files)    

# 4. master frame
# 4-1) create master bias
def create_master_bias(new_folder_path):
    # Path to the .list file
    list_file_path = os.path.join(new_folder_path, 'crop_bias.list')
    
    # Read file paths from the .list file
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]
    
    # Bias image stack
    bias_stack = []

    for file_path in file_paths:
        with fits.open(file_path) as hdul:
            bias_data = hdul[0].data
            bias_stack.append(bias_data)

    # Convert list to numpy array
    bias_stack = np.array(bias_stack)
    # Calculate the median across the stack
    master_bias = np.median(bias_stack, axis=0)
    
    return master_bias

# 4-2) create master dark
def create_master_dark(mbias, new_folder_path, exptime):
    # exptime : 300, 1800
    
    # Path to the .list file
    list_file_path = os.path.join(new_folder_path, f'crop_dark_{exptime}s.list')
    
    # Read file paths from the .list file
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]
    
    # Dark image stack
    dark_stack = []

    for file_path in file_paths:
        with fits.open(file_path) as hdul:
            dark_data = hdul[0].data
            corrected_dark = dark_data - mbias
            dark_stack.append(corrected_dark)

    # Convert list to numpy array
    dark_stack = np.array(dark_stack)
    # Calculate the median across the stack
    master_dark = np.median(dark_stack, axis=0)
    
    return master_dark

# 4-3) create master flat
def create_master_flat(mbias, new_folder_path):
    # Dark removal is omitted
    
    # Path to the .list file
    list_file_path = os.path.join(new_folder_path, f'crop_flat.list')
    
    # Read file paths from the .list file
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]
    
    # Flat image stack
    flat_stack = []

    for file_path in file_paths:
        with fits.open(file_path) as hdul:
            flat_data = hdul[0].data
            corrected_flat = flat_data - mbias
        
            # Find the mode at each flat
            flat_mode_result = stats.mode(corrected_flat, axis = None, keepdims = False)
            flat_mode = flat_mode_result.mode  # mode array data
            flat_mode = flat_mode[0] if isinstance(flat_mode, np.ndarray) else flat_mode # Choose the first mode
            
            # Normalize the flat frame by dividing by the mode
            normalized_flat = corrected_flat / flat_mode
            
            # Add the normalized flat frame to the stack
            flat_stack.append(normalized_flat)
    
    master_flat = np.median(flat_stack, axis=0)
    
    return master_flat
    
# 4-4) save fits image - single data
def save_mframe(master_data, new_folder_path, img_type):
    # img_type : bias, dark_300s, dark_1800s, flat
    output_file = os.path.join(new_folder_path, f'm{img_type}.fits')
    fits.writeto(output_file, master_data, overwrite=True)

# 5. preprocessing
def preprocess_img(mbias, mflat, img_type, new_folder_path):
    # img_type : piser, tcrb, arc
    
    # Path to the .list file
    list_file_path = os.path.join(new_folder_path, f'crop_{img_type}.list')
    
    # Read file paths from the .list file
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]

    preprocessed_data_list = []

    # Open the .list file and process each file path
    with open(list_file_path, 'r') as file:
        for line in file:
            # Strip whitespace and skip empty lines
            file_path = line.strip()

            # Open the FITS file
            with fits.open(file_path) as hdul:
                img = hdul[0].data  # Get the image data

                # Preprocess the image
                preprocessed_img = (img - mbias) / mflat

                # Append the file path and cropped data to the list
                preprocessed_data_list.append((file_path, preprocessed_img))

    return preprocessed_data_list
    
# 6. cosmic ray removal    
def cr_removal(img_type, new_folder_path, sigclip, sigfrac, gain, readnoise):
    # Path to the .list file
    list_file_path = os.path.join(new_folder_path, f'prepro_{img_type}.list')
    
    # Read file paths from the .list file
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]

    crr_data_list = []

    # Open the .list file and process each file path
    with open(list_file_path, 'r') as file:
        for line in file:
            # Strip whitespace and skip empty lines
            file_path = line.strip()

            # Open the FITS file
            with fits.open(file_path) as hdul:
                img = hdul[0].data  # Get the image data

                # Preprocess the image
                clean_spectrum = astroscrappy.detect_cosmics(
                    indat = img,
                    sigclip = sigclip,
                    sigfrac = sigfrac,
                    gain = gain,
                    readnoise = readnoise
                    )[1]

                # Append the file path and cropped data to the list
                crr_data_list.append((file_path, clean_spectrum))

    return crr_data_list

    
# 7. geometry
# 7-1) calculate the total rotation angle
def geometry_twice(new_folder_path):
    rotation_angle1 = geometry_angle(new_folder_path, reps = 0)
    rotation_angle2 = geometry_angle(new_folder_path, reps = 1)
    total_angle = rotation_angle1 + rotation_angle2
    
    return rotation_angle1, rotation_angle2, total_angle

def geometry_angle(new_folder_path, reps):
    # Select path for first image among flat images
    if reps == 0:
        flat_list_path = os.path.join(new_folder_path, 'crr_piser.list')
        flat_image_path = path_of_the_first_flat(flat_list_path)
        rotated_fits_path = os.path.join(new_folder_path, f'rot_piser.fits')
    elif reps == 1:
        flat_image_path = os.path.join(new_folder_path, f'rot_piser.fits')
        rotated_fits_path = os.path.join(new_folder_path, f'rot2_piser.fits')
    
    hdu = fits.open(flat_image_path)[0]    
    img = hdu.data
    
    height, width = img.shape
    column_indices = np.arange(0, width, 50)  # n픽셀 간격으로 열 선택(0에서 width까지 n간격으로)
    median_values = np.zeros(len(column_indices))  # 중앙값 저장할 array 만듦

    peak_positions = []
    x_data = np.arange(height)  # 마스킹된 이미지 내의 좌표

    for i, col_idx in enumerate(column_indices):
        # 각 열의 데이터
        y_data = img[:, col_idx]

        # 신호 세기가 너무 낮은 경우 스킵 (노이즈로 간주)
        if np.max(y_data) < 0.1 * np.max(img):
            peak_positions.append(np.nan)  # 너무 낮은 값은 무시 (결과에서 제외)
            continue

        # 가우시안의 초기 추정값 설정 (amp: 최대값, mu: 중앙값 근처, sigma: 약간 넓은 분포, offset: 최소값)
        amp_guess = np.max(y_data) - np.min(y_data)  # 신호의 범위
        mu_guess = np.argmax(y_data)  # 가장 큰 값의 위치
        sigma_guess = 10  # 분포의 넓이 추정
        offset_guess = np.min(y_data)  # 최소값을 오프셋으로 추정
        initial_guess = [amp_guess, mu_guess, sigma_guess, offset_guess]

        # curve fitting
        try:
            popt, _ = curve_fit(gaussian, x_data, y_data, p0 = initial_guess, maxfev = 10000)
            # 피크 위치는 마스킹한 이미지 기준이므로 원본 이미지 기준으로 보정
            peak_positions.append(popt[1])
        except RuntimeError:
            # 피팅 실패 시 이전 열의 피크값으로 대체
            if i > 0:
                peak_positions.append(peak_positions[-1])
            else:
                peak_positions.append(mu_guess)  # 첫 번째 열이면 초기 추정값 사용
                
    # nan 값이 없는 데이터만 추출
    valid_indices = ~np.isnan(peak_positions)  # nan이 아닌 값들의 인덱스
    valid_column_indices = column_indices[valid_indices]
    valid_peak_positions = np.array(peak_positions)[valid_indices]

    # 8. 선형 추세선 구하기
    slope, intercept = np.polyfit(valid_column_indices, valid_peak_positions, 1)  # 1차 다항식에 피팅

    # 10. 기울기(slope)로부터 회전 각도 계산
    rotation_angle = np.degrees(np.arctan(slope))
    
    rotated_image = rotate(img, rotation_angle, reshape = False, cval = 0)
    
    hdu_rotated = fits.PrimaryHDU(rotated_image)
    hdu_rotated.writeto(rotated_fits_path, overwrite = True)
    
    return rotation_angle

def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset

# 7-2) rotation all images
def geometry_img(img_type, new_folder_path, total_angle):
    # Path to the .list file
    list_file_path = os.path.join(new_folder_path, f'crr_{img_type}.list')
    
    # Read file paths from the .list file
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]

    geo_data_list = []

    # Open the .list file and process each file path
    with open(list_file_path, 'r') as file:
        for line in file:
            # Strip whitespace and skip empty lines
            file_path = line.strip()

            # Open the FITS file
            with fits.open(file_path) as hdul:
                img = hdul[0].data  # Get the image data

                # Rotate the image
                geometry_img = rotate(img, total_angle, reshape = False, order=5)
                # geometry_img = rotate(img, total_angle, reshape = False, order=5, mode='reflect')


                # Append the file path and rotated data to the list
                geo_data_list.append((file_path, geometry_img))

    return geo_data_list
    
    
# 8. extraction
# 8-1) y-coordinate at which the spectral line appears
# 가우시안 피팅으로 피크 찾아 중앙값으로 median_y 결정
# def extract_spectrum_y_coordi(img_type, new_folder_path, extract_n=5, extract_dW=4, plotCheck=False, plotCheck2=False):
#     list_file_path = os.path.join(new_folder_path, f'geo_{img_type}.list')
    
#     with open(list_file_path, 'r') as list_file:
#         file_paths = [line.strip() for line in list_file if line.strip()]

#     median_y_data = []
    
#     for file_path in file_paths:
#         with fits.open(file_path) as hdul:
#             img = hdul[0].data  # Get the image data
            
#             # X축을 extract_n개로 나누어 샘플링
#             x_positions = np.linspace(0, img.shape[1] - 1, extract_n, dtype=int)
#             peak_y_positions = []

#             for x in x_positions:
#                 peak_region = range(max(0, x - extract_dW), min(img.shape[1], x + extract_dW + 1))  # extract_dW 범위 내 x값들

#                 y_peaks = []
#                 for x_val in peak_region:
#                     y_values = img[:, x_val]  # 특정 x 값의 세로(열) 강도 데이터
#                     y_indices = np.arange(len(y_values))  # y 좌표들

#                     # 초기 추정값 (최대 강도 값 기준)
#                     amp_guess = np.max(y_values)  # 피크 높이
#                     mean_guess = np.argmax(y_values)  # 최대 강도 위치
#                     sigma_guess = 2  # 대략적인 분광선 너비
                    
#                     try:
#                         # 가우시안 피팅 수행
#                         popt, _ = curve_fit(gaussian, y_indices, y_values, 
#                                             p0=[amp_guess, mean_guess, sigma_guess, np.min(y_values)],
#                                             maxfev=5000)  # maxfev: 최대 반복 횟수 증가
#                         y_peak = popt[1]  # 피팅된 가우시안의 중심(y0)
#                     except:
#                         y_peak = mean_guess  # 피팅 실패 시 기본 최대값 사용
                    
#                     y_peaks.append(y_peak)

#                 # 여러 x 위치에서 얻은 y_peak 값들의 중앙값을 사용
#                 peak_y_positions.append(int(np.median(y_peaks)))

#             # 최종적으로 중앙값을 median_y로 설정
#             median_y = int(np.median(peak_y_positions))
#             median_y_data.append((file_path, median_y))

#             # 플롯 출력 (median_y를 이미지 위에 표시)
#             if plotCheck:
#                 plt.figure(figsize=(10, 5))
#                 plt.imshow(img, cmap='gray', origin='lower', aspect='auto')
#                 plt.axhline(y=median_y, color='red', linestyle='--', label=f"Median Y: {median_y}")
#                 plt.title(f'Image: {os.path.basename(file_path)} - Median Y: {median_y}')
#                 plt.legend()
#                 plt.show()

#             # 플롯 출력 (각 열에서 피팅된 y 피크 값들을 점으로 표시)
#             if plotCheck2:
#                 plt.figure(figsize=(10, 5))
#                 plt.imshow(img, cmap='gray', origin='lower', aspect='auto')
#                 plt.scatter(x_positions, peak_y_positions, color='cyan', label="Fitted Peak Positions")
#                 plt.axhline(y=median_y, color='red', linestyle='--', label=f"Median Y: {median_y}")
#                 plt.title(f'Fitted Spectrum Peak Positions for {os.path.basename(file_path)}')
#                 plt.legend()
#                 plt.show()

#     return median_y_data

# 가장 큰 합의 행이 median_y
def extract_spectrum_y_coordi(img_type, new_folder_path, plotCheck=False):
    list_file_path = os.path.join(new_folder_path, f'geo_{img_type}.list')
    
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]

    median_y_data = []
    
    for file_path in file_paths:
        with fits.open(file_path) as hdul:
            img = hdul[0].data  # Get the image data

            # 모든 열(column)을 합산하여 행(row)별 강도 계산
            row_intensity = np.sum(img, axis=1)  # 행 방향으로 합산
            median_y = int(np.argmax(row_intensity))  # 가장 높은 강도를 가지는 y 인덱스 찾기
            
            # 결과 저장
            median_y_data.append((file_path, median_y))

            # 플롯 출력
            if plotCheck:
                plt.figure(figsize=(10, 5))
                plt.imshow(img, cmap='gray', origin='lower', aspect='auto')
                plt.axhline(y=median_y, color='red', linestyle='--', label=f"Median Y: {median_y}")
                plt.title(f'Image: {os.path.basename(file_path)} - Median Y: {median_y}')
                plt.legend()
                plt.show()

    return median_y_data


# 8-2) Obj, Std extract spectrum value
def fwhm_gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def compute_fwhm(img_type, new_folder_path, median_y_data, y_limit=50, sigma=2, combine_row=1, 
                      plotCheck=False, plotCheck2=False, plotCheck_all=False):

    list_file_path = os.path.join(new_folder_path, f'geo_{img_type}.list')
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]

    fwhm_data = []

    for file_path, median_y in median_y_data:
        with fits.open(file_path) as hdul:
            img = hdul[0].data
            y_min, y_max = max(0, median_y - y_limit), min(img.shape[0], median_y + y_limit)
            y_indices = np.arange(y_min, y_max)
            img_roi = img[y_indices, :]

            # 기존 처리 방식
            full_cols = img_roi.shape[1] // combine_row
            remaining_cols = img_roi.shape[1] % combine_row  # 남는 열 개수

            # FWHM 계산을 위한 새로운 배열 크기 설정
            new_width = full_cols + (1 if remaining_cols > 0 else 0)
            combined_img = np.zeros((len(y_indices), new_width))

            # 기존 combine_row 처리된 부분
            if full_cols > 0:
                combined_img[:, :full_cols] = np.mean(
                    img_roi[:, :full_cols * combine_row].reshape(len(y_indices), full_cols, combine_row),
                    axis=2
                )
            
            # 남는 열 처리
            if remaining_cols > 0:
                combined_img[:, -1] = np.mean(img_roi[:, full_cols * combine_row:], axis=1)

            fwhm_values = np.full(new_width, None)
            mu_fit_values = np.full(new_width, None)

            for x in range(new_width):
                column_data = combined_img[:, x]
                smoothed_column = gaussian_filter1d(column_data, sigma=sigma)
                peak_idx, _ = find_peaks(smoothed_column)

                if len(peak_idx) == 0:
                    continue

                peak_y = peak_idx[np.argmax(smoothed_column[peak_idx])]
                fit_range = 7
                y_fit = y_indices[max(0, peak_y - fit_range): min(len(y_indices), peak_y + fit_range)]
                x_fit = smoothed_column[max(0, peak_y - fit_range): min(len(y_indices), peak_y + fit_range)]

                A_guess = np.max(x_fit)
                mu_guess = y_indices[peak_y]
                sigma_guess = np.std(y_fit)

                try:
                    popt, _ = curve_fit(fwhm_gaussian, y_fit, x_fit, p0=[A_guess, mu_guess, sigma_guess])
                    _, mu_fit, sigma_fit = popt
                    fwhm_values[x] = 2.355 * abs(sigma_fit)
                    mu_fit_values[x] = mu_fit
                except:
                    continue

            fwhm_data.append((file_path, fwhm_values))

            if plotCheck:
                plt.figure(figsize=(10, 6))
                plt.imshow(img, cmap='gray', origin='lower', aspect='auto')
                plt.axhline(y_min, color='cyan', linestyle='--', label='y_min limit')
                plt.axhline(y_max, color='cyan', linestyle='--', label='y_max limit')
                plt.title(f'FWHM Data Region: {os.path.basename(file_path)}')
                plt.xlabel('X (Column Index)')
                plt.ylabel('Y (Row Index)')
                plt.legend()
                plt.show()

            if plotCheck2:
                plt.figure(figsize=(10, 6))
                plt.imshow(img, cmap='gray', origin='lower', aspect='auto')
                valid_x = np.array([x for x, mu in enumerate(mu_fit_values) if mu is not None])
                valid_mu = np.array([mu for mu in mu_fit_values if mu is not None])
                if len(valid_x) > 0:
                    repeated_x = valid_x * combine_row + combine_row // 2
                    plt.scatter(repeated_x, valid_mu, color='blue', s=5, label='Gaussian Peak (μ)')
                plt.legend()
                plt.title(f'Gaussian Peak Visualization: {os.path.basename(file_path)}')
                plt.xlabel('X (Column Index)')
                plt.ylabel('Y (Row Index)')
                plt.show()

            if plotCheck_all:
                plt.figure(figsize=(10, 6))
                plt.imshow(img, cmap='gray', origin='lower', aspect='auto')
                first_label_mu = True
                first_label_fwhm = True
                for i, fwhm in enumerate(fwhm_values):
                    if mu_fit_values[i] is not None:
                        label_mu = 'Gaussian Peak (μ)' if first_label_mu else ""
                        plt.scatter(i * combine_row + combine_row // 2, mu_fit_values[i], color='blue', s=5, label=label_mu)
                        first_label_mu = False
                    if fwhm is not None and mu_fit_values[i] is not None:
                        label_fwhm = 'FWHM Range' if first_label_fwhm else ""
                        plt.plot([
                            i * combine_row, i * combine_row
                        ], [
                            mu_fit_values[i] - fwhm / 2, mu_fit_values[i] + fwhm / 2
                        ], color='red', alpha=0.3, linewidth=1.5, label=label_fwhm)
                        first_label_fwhm = False
                plt.legend()
                plt.title(f'FWHM Visualization: {os.path.basename(file_path)}')
                plt.xlabel('X (Column Index)')
                plt.ylabel('Y (Row Index)')
                plt.show()

    return fwhm_data


def compute_background(img_type, new_folder_path, median_y_data, fwhm_data, fwhm_inner=1.0, fwhm_outer=2.0, plotCheck=False):
    list_file_path = os.path.join(new_folder_path, f'geo_{img_type}.list')
    
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]
    
    background_values = []
    
    for (file_path, median_y), (_, segmented_fwhm) in zip(median_y_data, fwhm_data):
        with fits.open(file_path) as hdul:
            img = hdul[0].data  # Get the image data
            segment_size = img.shape[1] // len(segmented_fwhm)
            background_column_values = []
            
            for x in range(img.shape[1]):
                segment_idx = min(x // segment_size, len(segmented_fwhm) - 1)
                fwhm_rep = segmented_fwhm[segment_idx]
                
                if fwhm_rep is None:
                    background_column_values.append(None)
                    continue
                
                lower_inner = round(median_y - (fwhm_inner / 2) * fwhm_rep)
                lower_outer = round(median_y - (fwhm_outer / 2) * fwhm_rep)
                upper_inner = round(median_y + (fwhm_inner / 2) * fwhm_rep)
                upper_outer = round(median_y + (fwhm_outer / 2) * fwhm_rep)
                
                lower_region = img[max(0, lower_outer):max(0, lower_inner), x]
                upper_region = img[min(img.shape[0], upper_inner):min(img.shape[0], upper_outer), x]
                
                lower_median = np.median(lower_region) if lower_region.size > 0 else None
                upper_median = np.median(upper_region) if upper_region.size > 0 else None
                
                if lower_median is not None and upper_median is not None:
                    background_value = (lower_median + upper_median) / 2
                elif lower_median is not None:
                    background_value = lower_median
                elif upper_median is not None:
                    background_value = upper_median
                else:
                    background_value = None
                
                background_column_values.append(background_value)
            
            background_values.append((file_path, background_column_values))
            
            # Plot background values
            if plotCheck:
                plt.figure(figsize=(10, 5))
                plt.plot(range(img.shape[1]), background_column_values, label='Background Values', color='blue')
                plt.xlabel('Column Index')
                plt.ylabel('Background Value')
                plt.title(f'Background Values for {os.path.basename(file_path)}')
                plt.legend()
                plt.grid()
                plt.show()    
    
    return background_values


# 배경을 빼기 전과 후의 스펙트럼 값을 함께 보여주는 함수
def compute_spectrum(img_type, new_folder_path, median_y_data, fwhm_data, background_values, fwhm_spectrum=4, plotCheck=False, plotCheck2=False):
    list_file_path = os.path.join(new_folder_path, f'geo_{img_type}.list')
    
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]
    
    spectrum_values = []

    for (file_path, median_y), (_, segmented_fwhm), (_, background_column_values) in zip(median_y_data, fwhm_data, background_values):
        with fits.open(file_path) as hdul:
            img = hdul[0].data  # Get the image data
            segment_size = img.shape[1] // len(segmented_fwhm)
            raw_spectrum_values = []
            corrected_spectrum_values = []

            for x in range(img.shape[1]):
                segment_idx = min(x // segment_size, len(segmented_fwhm) - 1)
                fwhm_rep = segmented_fwhm[segment_idx] # 해당하는 구간의 구간별 FWHM

                if fwhm_rep is None or background_column_values[x] is None:
                    raw_spectrum_values.append(None)
                    corrected_spectrum_values.append(None)
                    continue

                # 스펙트럼 추출할 구간 설정 (총 길이 : fwhm_spectrum * fwhm_rep)
                lower_bound = round(median_y - (fwhm_spectrum / 2) * fwhm_rep)
                upper_bound = round(median_y + (fwhm_spectrum / 2) * fwhm_rep)

                # 이미지를 벗어나지 않도록 조정
                lower_bound = max(0, lower_bound)
                upper_bound = min(img.shape[0], upper_bound)

                # 해당 열에서 스펙트럼 값 계산 (구간 내 데이터 합)
                spectrum_value = np.sum(img[lower_bound:upper_bound, x])

                # 배경 보정: {배경값 * fwhm_spectrum} 길이만큼 빼줌
                background_correction = background_column_values[x] * (upper_bound - lower_bound)
                corrected_spectrum_value = spectrum_value - background_correction

                raw_spectrum_values.append(spectrum_value)
                corrected_spectrum_values.append(corrected_spectrum_value)
                
            spectrum_values.append((file_path, corrected_spectrum_values))

            # 그래프 출력
            if plotCheck: 
                plt.figure(figsize=(10, 5))
                plt.plot(range(img.shape[1]), raw_spectrum_values, label='Raw Spectrum', color='blue', alpha=0.6)
                plt.plot(range(img.shape[1]), corrected_spectrum_values, label='Background Subtracted Spectrum', color='red')
                plt.xlabel('Column Index')
                plt.ylabel('Spectrum Value')
                plt.title(f'Spectrum for {os.path.basename(file_path)}')
                plt.legend()
                plt.grid()
                plt.show()

            # 이미지 위에 FWHM 추출 구간 표시
            if plotCheck2:
                plt.figure(figsize=(10, 5))
                plt.imshow(img, cmap='gray', origin='lower', aspect='auto')
                
                for x in range(img.shape[1]):
                    segment_idx = min(x // segment_size, len(segmented_fwhm) - 1)
                    fwhm_rep = segmented_fwhm[segment_idx]

                    if fwhm_rep is None:
                        continue
                    
                    lower_bound = int(median_y - (fwhm_spectrum / 2) * fwhm_rep)
                    upper_bound = int(median_y + (fwhm_spectrum / 2) * fwhm_rep)
                    lower_bound = max(0, lower_bound)
                    upper_bound = min(img.shape[0], upper_bound)
                    
                    plt.plot([x, x], [lower_bound, upper_bound], color='yellow', alpha=0.1)  # 추출 구간 표시
                
                plt.xlabel('Column Index')
                plt.ylabel('Row Index')
                plt.title(f'Spectrum Extraction Region for {os.path.basename(file_path)}')
                plt.show()
                
    return spectrum_values


# 임의로 양쪽에서 일부 데이터 지움 - 양쪽에서 각각 delete_data 픽셀만큼
def trim_spectrum_data(spectrum_values, delete_data, plotCheck=False, compare=False):
    trimmed_spectrum_values = []
    
    for file_path, spectrum_data in spectrum_values:
        if len(spectrum_data) <= 2 * delete_data:
            trimmed_data = []  # If too few data points remain, return an empty list
        else:
            trimmed_data = spectrum_data[delete_data:-delete_data]
        
        trimmed_spectrum_values.append((file_path, trimmed_data))
        
        # Plot trimmed spectrum values
        if plotCheck: 
            plt.figure(figsize=(10, 5))
            if compare:
                plt.plot(range(len(spectrum_data)), spectrum_data, label='Original Spectrum', color='blue', alpha=0.6)
            plt.plot(range(delete_data, len(spectrum_data) - delete_data), trimmed_data, label='Trimmed Spectrum', color='red')
            plt.xlabel('Column Index')
            plt.ylabel('Spectrum Value')
            plt.title(f'Trimmed Spectrum for {os.path.basename(file_path)}')
            plt.legend()
            plt.grid()
            plt.show()
    
    return trimmed_spectrum_values


# 8-3) Arc extract spectrum value
def extract_arc_spectrum(img_type, new_folder_path, median_y_data, fwhm_data, fwhm_arc_spectrum=4, plotCheck=False):
    arc_list_path = os.path.join(new_folder_path, 'geo_arc.list')
    
    with open(arc_list_path, 'r') as arc_file:
        arc_file_paths = [line.strip() for line in arc_file if line.strip()]
    
    extracted_spectrum_list = []
    total_processed = 0
    processing_summary = []
    
    for arc_file_path in arc_file_paths:
        with fits.open(arc_file_path) as hdul:
            arc_img = hdul[0].data 
            applied_files = []
            
            for (star_file, median_y), (_, segmented_fwhm) in zip(median_y_data, fwhm_data):
                segment_size = arc_img.shape[1] // len(segmented_fwhm)
                extracted_spectrum = []
                
                for x in range(arc_img.shape[1]):
                    segment_idx = min(x // segment_size, len(segmented_fwhm) - 1)
                    fwhm_rep = segmented_fwhm[segment_idx]
                    
                    if fwhm_rep is None:
                        extracted_spectrum.append(None)
                        continue
                    
                    lower_bound = round(median_y - (fwhm_arc_spectrum / 2) * fwhm_rep)
                    upper_bound = round(median_y + (fwhm_arc_spectrum / 2) * fwhm_rep)
                    
                    lower_bound = max(0, lower_bound)
                    upper_bound = min(arc_img.shape[0], upper_bound)
                    
                    spectrum_value = np.sum(arc_img[lower_bound:upper_bound, x])
                    extracted_spectrum.append(spectrum_value)
                    
                extracted_spectrum_list.append((arc_file_path, star_file, extracted_spectrum))
                total_processed += 1
                applied_files.append(os.path.basename(star_file))
                
                # 그래프 출력
                if plotCheck:
                    plt.figure(figsize=(10, 5))
                    plt.plot(range(len(extracted_spectrum)), extracted_spectrum, label='Extracted Spectrum', color='blue')
                    plt.xlabel('Column Index')
                    plt.ylabel('Spectrum Value')
                    plt.title(f'Spectrum for {os.path.basename(star_file)} applied to {os.path.basename(arc_file_path)}')
                    plt.legend()
                    plt.grid()
                    plt.show()
                
            processing_summary.append(f"{os.path.basename(arc_file_path)}에는 {', '.join(applied_files)}가 적용되어 처리되었습니다.")
    
    print("\n".join(processing_summary))
    print(f"A total of {total_processed} processing steps were completed in '{img_type}' process.")
    return extracted_spectrum_list

def trim_arc_data(extracted_spectrum_list, delete_data, plotCheck=False, compare=False):
    trimmed_spectrum_list = []
    
    for arc_file_path, star_file, spectrum_data in extracted_spectrum_list:
        if len(spectrum_data) <= 2 * delete_data:
            trimmed_data = []  # 남은 데이터가 너무 적으면 빈 리스트 반환
        else:
            trimmed_data = spectrum_data[delete_data:-delete_data]
        
        trimmed_spectrum_list.append((arc_file_path, star_file, trimmed_data))
        
        # 그래프 출력
        if plotCheck:
            plt.figure(figsize=(10, 5))
            if compare:
                plt.plot(range(len(spectrum_data)), spectrum_data, label='Original Spectrum', color='blue', alpha=0.6)
            plt.plot(range(delete_data, len(spectrum_data) - delete_data), trimmed_data, label='Trimmed Spectrum', color='red')
            plt.xlabel('Column Index')
            plt.ylabel('Spectrum Value')
            plt.title(f'Trimmed Spectrum for {os.path.basename(star_file)} applied to {os.path.basename(arc_file_path)}')
            plt.legend()
            plt.grid()
            plt.show()
    
    return trimmed_spectrum_list


def save_spectrum_data_to_FITS(img_type, new_folder_path, spectrum_values, extracted_spectrum_list):
    spectrum_files = []
    
    # 스펙트럼 데이터 저장
    for file_path, spectrum in spectrum_values:
        file_name = os.path.basename(file_path)
        fits_file_path = os.path.join(new_folder_path, f'extrac_{file_name}.fits')
        
        # FITS 헤더 및 데이터 생성
        hdu = fits.PrimaryHDU(np.array(spectrum))
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_file_path, overwrite=True)
        
        print(f"Spectrum saved to {fits_file_path}")
        spectrum_files.append(fits_file_path)
    
    # 아크 데이터 저장
    arc_files_dict = {}
    for arc_file, star_file, extracted_spectrum in extracted_spectrum_list:
        arc_name = os.path.basename(arc_file)
        star_name = os.path.basename(star_file)
        fits_file_path = os.path.join(new_folder_path, f'extrac_{arc_name}_{star_name}.fits')
        
        # FITS 헤더 및 데이터 생성
        hdu = fits.PrimaryHDU(np.array(extracted_spectrum))
        hdul = fits.HDUList([hdu])
        hdul.writeto(fits_file_path, overwrite=True)
        
        print(f"Extracted spectrum saved to {fits_file_path}")
        
        # 아크별로 파일을 정리
        if arc_name not in arc_files_dict:
            arc_files_dict[arc_name] = []
        arc_files_dict[arc_name].append(fits_file_path)
    
    # 아크 파일 리스트 저장 (img_type별로 저장)
    arc_count = 0
    for arc_name, files in arc_files_dict.items():
        arc_count += 1
        arc_list_path = os.path.join(new_folder_path, f'extrac_arc{arc_count}_{img_type}.list')
        with open(arc_list_path, 'w') as f:
            f.write('\n'.join(files) + '\n')
        print(f"Arc file list saved to {arc_list_path}")
    
    # 스펙트럼 파일 리스트 저장
    spectrum_list_path = os.path.join(new_folder_path, f'extrac_{img_type}.list')
    with open(spectrum_list_path, 'w') as f:
        f.write('\n'.join(spectrum_files) + '\n')
    print(f"Spectrum file list saved to {spectrum_list_path}")



# 9. wavelength calibration
# 9-1) Relationship between pixels and wavelengths
# 여기부터 3개 함수 : 4개로 일차식 피팅팅
# def linear_fit(x, a, b):
#     return a * x + b

# def plot_spectra_from_list(img_type, new_folder_path, plotCheck=False):
#     list_files = sorted(glob.glob(os.path.join(new_folder_path, f'extrac_arc*_{img_type}.list')),
#                          key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    
#     if not list_files:
#         print(f"No list files found for img_type {img_type} in {new_folder_path}.")
#         return
    
#     pixel_positions = []
#     wavelength_positions = []
    
#     for list_file_path in list_files:
#         with open(list_file_path, 'r') as f:
#             fits_files = [line.strip() for line in f.readlines() if line.strip()]
        
#         for fits_file in fits_files:
#             peak_indices = plot_spectrum_with_peaks(fits_file, plotCheck)
            
#             if len(peak_indices) == 4:
#                 pixel_positions.extend(peak_indices)
#                 wavelength_positions.extend([5852.488, 6143.063, 6402.246, 6678.276])
                
#                 if len(pixel_positions) > 3:
#                     popt, _ = curve_fit(linear_fit, pixel_positions, wavelength_positions)
#                     print(f"Fitted linear equation: Wavelength = {popt[0]:.6f} * Pixel + {popt[1]:.6f}")
#                 else:
#                     print("Not enough data points for linear fitting yet.")

#     return popt[0], popt[1]

# def plot_spectrum_with_peaks(fits_file, plotCheck, sigma=1):
#     # FITS 파일 로드
#     with fits.open(fits_file) as hdul:
#         spectrum_data = hdul[0].data
    
#     # 가우시안 스무딩 적용
#     smoothed_spectrum = gaussian_filter1d(spectrum_data, sigma=sigma)
    
#     # 극댓값 찾기
#     peaks, _ = find_peaks(smoothed_spectrum)
    
#     # 가장 큰 피크 선택 (가장 높은 피크를 기준으로 오른쪽에서 3개 더 찾음)
#     highest_peak = max(peaks, key=lambda x: smoothed_spectrum[x])
#     right_peaks = [p for p in peaks if p > highest_peak]
#     sorted_right_peaks = sorted(right_peaks, key=lambda x: smoothed_spectrum[x], reverse=True)
#     top_peaks = [highest_peak] + sorted(sorted_right_peaks[:3])  # 기준점 + 오른쪽에서 3개
    
#     if len(top_peaks) < 4:
#         print(f"Warning: Less than 4 peaks found in {fits_file}")
#         return []
    
#     peak_wavelengths = [5852.488, 6143.063, 6402.246, 6678.276]
    
#     # 플로팅
#     if plotCheck:
#         plt.figure(figsize=(8, 5))
#         plt.plot(smoothed_spectrum, label='Smoothed Spectrum')
        
#         for i, (idx, wavelength) in enumerate(zip(top_peaks, peak_wavelengths)):
#             plt.axvline(x=idx, color='r', linestyle='dashed', label=f'Peak {i+1} at {idx}')
#             plt.text(idx, smoothed_spectrum[idx], f'{idx}\n{wavelength:.3f}', color='red', fontsize=12, verticalalignment='bottom')
        
#         plt.xlabel('Pixel')
#         plt.ylabel('Intensity')
#         plt.title(f'Peak Detection in {os.path.basename(fits_file)}')
#         plt.legend()
#         plt.grid()
#         plt.show()
    
#     return top_peaks  # 극댓값 위치 반환

# 1개로 일차식 피팅(slope는 주고 상수항만 계산)
def calculate_intercept(img_type, new_folder_path, slope = 0.9, sigma=1, plotCheck=False):  
    list_files = sorted(glob.glob(os.path.join(new_folder_path, f'extrac_arc*_{img_type}.list')),
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    
    for list_file_path in list_files:
        with open(list_file_path, 'r') as f:
            fits_files = [line.strip() for line in f.readlines() if line.strip()]
        
        for fits_file in fits_files:
            with fits.open(fits_file) as hdul:
                spectrum_data = hdul[0].data
            
            smoothed_spectrum = gaussian_filter1d(spectrum_data, sigma=sigma)
            peaks, _ = find_peaks(smoothed_spectrum)
            
            if not peaks.any():
                print(f"No peaks found in {fits_file}")
                continue
            
            highest_peak = max(peaks, key=lambda x: smoothed_spectrum[x])
            reference_wavelength = 5852.488  
            intercept = reference_wavelength - slope * highest_peak
            
            if plotCheck:
                plt.figure(figsize=(8, 5))
                plt.plot(smoothed_spectrum, label='Smoothed Spectrum')
                plt.axvline(x=highest_peak, color='r', linestyle='dashed', label=f'Highest Peak at {highest_peak}')
                plt.text(highest_peak, smoothed_spectrum[highest_peak] - 0.1 * max(smoothed_spectrum), 
                         f'{highest_peak}\n{reference_wavelength:.3f}', 
                         color='red', fontsize=12, verticalalignment='top')
                plt.xlabel('Pixel')
                plt.ylabel('Intensity')
                plt.title(f'Highest Peak Detection in {os.path.basename(fits_file)}')
                plt.legend()
                plt.grid()
                plt.show()
            
                print(f"Calculated intercept: {intercept:.6f}")
    
    return slope, intercept


# def trans_wavelength_positions(img_type, new_folder_path, wavelength_list, a, b, all_peaks, plotCheck=False):
    list_files = sorted(glob.glob(os.path.join(new_folder_path, f'extrac_arc*_{img_type}.list')),
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

    if not list_files:
        print(f"No list files found for img_type {img_type} in {new_folder_path}.")
        return

    fits_files = []
    for list_file_path in list_files:
        with open(list_file_path, 'r') as f:
            fits_files.extend([line.strip() for line in f.readlines() if line.strip()])

    if not fits_files:
        print("No FITS files found in the list files.")
        return

    # 각 이미지에 대한 peaks를 저장할 리스트
    knownwavelength_to_pixel = []

    # 각 이미지에 대한 peaks에 대해 처리
    for i, fits_file in enumerate(fits_files):
        # FITS 파일 로드
        with fits.open(fits_file) as hdul:
            spectrum_data = hdul[0].data

        # 파장값에 대한 픽셀 위치 계산
        pixel_positions = [(wavelength - b) / a for wavelength in wavelength_list]
        pixel_positions = np.round(pixel_positions)

        # 현재 이미지에 대한 peaks 가져오기
        peaks = all_peaks[i]

        # 스펙트럼 데이터 플로팅
        if plotCheck:
            plt.figure(figsize=(10, 5))
            plt.plot(spectrum_data, label='Original Spectrum')

            # 각 파장에 해당하는 픽셀 위치 표시
            for wave, pixel in zip(wavelength_list, pixel_positions):
                plt.axvline(x=pixel, color='r', linestyle='dashed', alpha=0.7)
                plt.text(pixel, max(spectrum_data) * 0.9, f'{wave:.1f} Å',
                         color='red', fontsize=10, rotation=90, verticalalignment='bottom')

            # 피크 위치 표시
            plt.scatter(peaks, spectrum_data[peaks], color='orange', marker='o', label='Detected Peaks')

            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.title(f'Wavelength Positions in {os.path.basename(fits_file)}')
            plt.legend()
            plt.grid()
            plt.show()

        # 이미지에 대한 피크 위치를 업데이트하여 저장
        knownwavelength_to_pixel.append(peaks)

    return knownwavelength_to_pixel
def trans_wavelength_positions(img_type, new_folder_path, wavelength_list, a, b, all_peaks, plotCheck=False):
    list_files = sorted(glob.glob(os.path.join(new_folder_path, f'extrac_arc*_{img_type}.list')),
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))) )

    if not list_files:
        print(f"No list files found for img_type {img_type} in {new_folder_path}.")
        return []

    fits_files = []
    for list_file_path in list_files:
        with open(list_file_path, 'r') as f:
            fits_files.extend([line.strip() for line in f.readlines() if line.strip()])

    if not fits_files:
        print("No FITS files found in the list files.")
        return []

    # 변환된 픽셀 위치를 저장할 리스트
    knownwavelength_to_pixel = []

    # 각 이미지 처리
    for i, fits_file in enumerate(fits_files):
        # FITS 파일 로드
        with fits.open(fits_file) as hdul:
            spectrum_data = hdul[0].data

        # 파장값에 대한 픽셀 위치 변환
        pixel_positions = np.round((np.array(wavelength_list) - b) / a).astype(int)
        
        # 변환된 값 저장 (peaks가 아니라 pixel_positions를 저장해야 함)
        knownwavelength_to_pixel.append(list(pixel_positions))
        # print(knownwavelength_to_pixel)

        # 스펙트럼 데이터 플로팅 (plotCheck가 True일 때)
        if plotCheck:
            plt.figure(figsize=(10, 5))
            plt.plot(spectrum_data, label='Original Spectrum')

            # 변환된 픽셀 위치 표시
            for wave, pixel in zip(wavelength_list, pixel_positions):
                plt.axvline(x=pixel, color='r', linestyle='dashed', alpha=0.7)
                plt.text(pixel, max(spectrum_data) * 0.9, f'{wave:.1f} Å',
                         color='red', fontsize=10, rotation=90, verticalalignment='bottom')

            # 검출된 피크 표시
            plt.scatter(all_peaks[i], spectrum_data[all_peaks[i]], color='orange', marker='o', label='Detected Peaks')

            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.title(f'Wavelength Positions in {os.path.basename(fits_file)}')
            plt.legend()
            plt.grid()
            plt.show()

    return knownwavelength_to_pixel


def detect_many_peaks(img_type, new_folder_path, height_factor=0.05, min_distance=10, plotCheck=False):
    list_files = sorted(glob.glob(os.path.join(new_folder_path, f'extrac_arc*_{img_type}.list')),
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

    if not list_files:
        print(f"No list files found for img_type {img_type} in {new_folder_path}.")
        return []

    fits_files = []
    for list_file_path in list_files:
        with open(list_file_path, 'r') as f:
            fits_files.extend([line.strip() for line in f.readlines() if line.strip()])

    if not fits_files:
        print("No FITS files found in the list files.")
        return []

    all_peaks = []  # 모든 이미지에 대한 피크를 저장할 리스트

    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            spectrum_data = hdul[0].data

        # 피크 검출
        peak_threshold = max(spectrum_data) * height_factor
        peaks, _ = find_peaks(spectrum_data, height=peak_threshold, distance=min_distance)

        # 이미지별 피크 위치 저장
        all_peaks.append(peaks)

        # 결과 플로팅
        if plotCheck:
            plt.figure(figsize=(10, 5))
            plt.plot(spectrum_data, label='Original Spectrum')
            plt.scatter(peaks, spectrum_data[peaks], color='orange', marker='o', label='Detected Peaks')

            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.title(f'All Detected Peaks in {os.path.basename(fits_file)}')
            plt.legend()
            plt.grid()
            plt.show()

    return all_peaks  # 모든 이미지의 피크를 반환



# 9-2) 파장 재보정
# 1차 매칭 : 알려진 파장의 픽셀 위치에서 
def adjust_pixel_positions_with_peaks(img_type, new_folder_path, knownwavelength_to_pixel, all_peaks, tolerance=10, plotCheck=False):
    adjusted_pixel_positions = []

    # 각 이미지에 대해 피크 정정
    for i, (img_known_positions, peaks) in enumerate(zip(knownwavelength_to_pixel, all_peaks)):
        adjusted_img_pixel_positions = []

        for pixel in img_known_positions:
            min_distance = float('inf')
            closest_peak = None

            # tolerance 범위 내에서 가장 가까운 피크 찾기
            for peak in peaks:
                distance = abs(pixel - peak)
                if distance <= tolerance and distance < min_distance:
                    closest_peak = peak
                    min_distance = distance

            # 가장 가까운 피크를 저장, 없으면 None
            adjusted_img_pixel_positions.append(closest_peak if closest_peak is not None else None)
        
        adjusted_pixel_positions.append(adjusted_img_pixel_positions)

    # 파일 목록 로드
    list_files = sorted(glob.glob(os.path.join(new_folder_path, f'extrac_arc*_{img_type}.list')),
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))) )
    if not list_files:
        print(f"No list files found for img_type {img_type} in {new_folder_path}.")
        return adjusted_pixel_positions

    fits_files = []
    for list_file_path in list_files:
        with open(list_file_path, 'r') as f:
            fits_files.extend([line.strip() for line in f.readlines() if line.strip()])
    
    if not fits_files:
        print("No FITS files found in the list files.")
        return adjusted_pixel_positions
    
    # 각 이미지별 스펙트럼 플로팅
    if plotCheck:
        for i, fits_file in enumerate(fits_files):
            with fits.open(fits_file) as hdul:
                spectrum_data = hdul[0].data

            plt.figure(figsize=(10, 5))
            plt.plot(spectrum_data, label='Extracted Arc Spectrum')
            
            # 현재 이미지에 대한 피크 위치 표시
            peaks = all_peaks[i]
            
            # 원본 파장 값에 해당하는 픽셀 위치 표시 (파란색)
            for pixel in knownwavelength_to_pixel[i]:
                plt.axvline(x=pixel, color='b', linestyle='dashed', alpha=0.7)
            
            # 정정된 픽셀 위치 표시 (빨간색)
            for pixel, adjusted_pixel in zip(knownwavelength_to_pixel[i], adjusted_pixel_positions[i]):
                if adjusted_pixel is not None:
                    plt.axvline(x=adjusted_pixel, color='r', linestyle='dashed', alpha=0.7)
            
            # legend
            plt.plot([], [], color='b', linestyle='dashed', label='wavelength (by linear expression)')
            plt.plot([], [], color='r', linestyle='dashed', label='wavelength (by nearest peak)')
            plt.scatter([], [], color='orange', marker='o', label='Detected Peaks')  # 피크 범례
            
            # 피크 위치 표시
            plt.scatter(peaks, spectrum_data[peaks], color='orange', marker='o')
            
            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.title(f'Pixel Positions in {os.path.basename(fits_file)}')
            plt.legend()
            plt.grid()
            plt.show()
            
            print(f"Adjusted Pixel Positions for {os.path.basename(fits_file)}: {adjusted_pixel_positions[i]}")

    return adjusted_pixel_positions

# 2차 매칭 : adjust된 파장의 부근에서 가우시안 --> 정확한 뮤값 찾고 매칭 
def refine_pixel_positions(img_type, new_folder_path, adjusted_pixel_positions, search_range=20, plotCheck=False):
    readjusted_pixel_positions = []

    # 파일 목록 로드
    list_files = sorted(glob.glob(os.path.join(new_folder_path, f'extrac_arc*_{img_type}.list')),
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    if not list_files:
        print(f"No list files found for img_type {img_type} in {new_folder_path}.")
        return adjusted_pixel_positions

    fits_files = []
    for list_file_path in list_files:
        with open(list_file_path, 'r') as f:
            fits_files.extend([line.strip() for line in f.readlines() if line.strip()])
    
    if not fits_files:
        print("No FITS files found in the list files.")
        return adjusted_pixel_positions
    
    for i, fits_file in enumerate(fits_files):
        with fits.open(fits_file) as hdul:
            spectrum_data = hdul[0].data
        
        refined_positions = []
        for adj_pixel in adjusted_pixel_positions[i]:
            if adj_pixel is None:
                refined_positions.append(None)
                continue
            
            x_range = np.arange(max(0, adj_pixel - search_range), min(len(spectrum_data), adj_pixel + search_range))
            y_range = spectrum_data[x_range]
            peak_height = np.max(y_range)
            
            # 초기 추정값: peak 값 근처에서 가우시안 피팅
            p0 = [peak_height, adj_pixel, 5, np.min(y_range)]
            try:
                popt, _ = curve_fit(gaussian, x_range, y_range, p0=p0)
                mu, sigma = popt[1], popt[2]
                refined_positions.append(mu)
            except RuntimeError:
                refined_positions.append(adj_pixel)  # 피팅 실패 시 기존 값 유지
        
        readjusted_pixel_positions.append(refined_positions)

        # 스펙트럼 및 조정된 위치 시각화 (plotCheck 활성화 시)
        if plotCheck:
            plt.figure(figsize=(10, 5))
            plt.plot(spectrum_data, label='Extracted Arc Spectrum')
            
            for adj_pixel in adjusted_pixel_positions[i]:
                if adj_pixel is not None:
                    x_min, x_max = adj_pixel - search_range, adj_pixel + search_range
                    y_min, y_max = 0, np.max(spectrum_data[x_min:x_max])
                    plt.fill_betweenx([y_min, y_max], x_min, x_max, color='yellow', alpha=0.5)
                    plt.axvline(x=adj_pixel, color='b', linestyle='dashed', alpha=0.7)
            
            for readj_pixel in refined_positions:
                if readj_pixel is not None:
                    plt.axvline(x=readj_pixel, color='springgreen', linestyle='dashed', alpha=0.7)
            
            # legend
            plt.plot([], [], color='b', linestyle='dashed', label='Adjusted Pixel')
            plt.plot([], [], color='springgreen', linestyle='dashed', label='ReAdjusted Pixel')
            plt.plot([], [], color='yellow', alpha=0.5, label='Gaussian Range')

            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.title(f'Pixel Refinement in {os.path.basename(fits_file)}')
            plt.legend()
            plt.grid()
            plt.show()

            print(f"ReAdjusted Pixel Positions for {os.path.basename(fits_file)}: {readjusted_pixel_positions[i]}")
    
    return readjusted_pixel_positions


def pair_wavelengths_with_pixels(known_wavelength_list, readjusted_pixel_positions):
    paired_results = []
    
    for pixel_positions in readjusted_pixel_positions:
        paired_results.append(list(zip(known_wavelength_list, pixel_positions)))
    
    return paired_results

# '처음 주어진 파장' - '2차 매칭 후 픽셀' 쌍들을 이용해 다항식 피팅
def fit_wavelength_pixel(multi_image_data, degree=2, result_all_polyfit=False, plotCheck_error=False):
    all_coeffs = []
    errors = []
    
    for img_idx, data in enumerate(multi_image_data):
        filtered_data = [(p, w) for p, w in data if w is not None]
        pixels, wavelengths = zip(*filtered_data)
        pixels = np.array(pixels)
        wavelengths = np.array(wavelengths)
        
        print(f"이미지 {img_idx + 1}:")
        
        if result_all_polyfit:
            coeffs_dict = {}
            for d in range(1, 6):
                coeffs = np.polyfit(wavelengths, pixels, d)
                poly_eq = " + ".join(f"{c:.6e}*x**{i}" for i, c in enumerate(reversed(coeffs)))
                print(f"  {d}차식: {poly_eq}")
                coeffs_dict[f"{d}차"] = coeffs.tolist()
            all_coeffs.append(coeffs_dict)
        else:
            coeffs = np.polyfit(wavelengths, pixels, degree)
            poly_eq = " + ".join(f"{c:.6e}*x**{i}" for i, c in enumerate(reversed(coeffs)))
            print(f"  {degree}차식: {poly_eq}")
            all_coeffs.append(coeffs.tolist())
            
            fitted_pixels = np.polyval(coeffs, wavelengths)
            error = pixels - fitted_pixels
            errors.append(error)
            
            if plotCheck_error:
                plt.figure(figsize=(8, 5))
                plt.scatter(wavelengths, error, label='Residuals', color='r', alpha=0.7)
                plt.axhline(0, color='black', linestyle='--')
                plt.xlabel('X-index (Pixel)')
                plt.ylabel('Residuals (Pixel)')
                plt.title(f'Fit Residuals for Image {img_idx + 1}')
                plt.legend()
                plt.grid()
                plt.show()
    
    return all_coeffs, errors

def convert_pixel_to_wavelength(img_type, new_folder_path, coeffs_list):
    list_files = sorted(glob.glob(os.path.join(new_folder_path, f'extrac_arc*_{img_type}.list')),
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    if not list_files:
        print(f"No list files found for img_type {img_type} in {new_folder_path}.")
        return [], []
    
    fits_files = []
    for list_file_path in list_files:
        with open(list_file_path, 'r') as f:
            fits_files.extend([line.strip() for line in f.readlines() if line.strip()])
    
    if not fits_files:
        print("No FITS files found in the list files.")
        return [], []
    
    wavelength_positions_list = []
    spectrum_data_list = []
    
    for fits_file, coeffs in zip(fits_files, coeffs_list):
        with fits.open(fits_file) as hdul:
            spectrum_data = hdul[0].data
        pixel_positions = np.arange(len(spectrum_data))
        wavelength_positions = np.polyval(coeffs, pixel_positions)
        
        wavelength_positions_list.append(wavelength_positions)
        spectrum_data_list.append(spectrum_data)
    
    return wavelength_positions_list, spectrum_data_list

def calculate_spectral_resolution(
    img_type,
    new_folder_path,
    coeffs_list,
    adjusted_pixel_positions_list,
    search_range=20,
    plotCheck_resolution=False,
    plotCheck_sigma=False
):
    wavelength_positions_list, spectrum_data_list = convert_pixel_to_wavelength(img_type, new_folder_path, coeffs_list)
    
    resolutions_list = []
    sigmas_list = []
    wavelengths_used_list = []

    for coeffs, wavelength_positions, spectrum_data, adjusted_pixel_positions in zip(coeffs_list, wavelength_positions_list, spectrum_data_list, adjusted_pixel_positions_list):
        if wavelength_positions is None or spectrum_data is None:
            resolutions_list.append([])
            sigmas_list.append([])
            wavelengths_used_list.append([])
            continue
        
        resolutions = []
        sigmas = []
        wavelengths_used = []
        for pixel in adjusted_pixel_positions:
            if pixel is None:
                resolutions.append(None)
                sigmas.append(None)
                continue
            
            wavelength = np.polyval(coeffs, pixel)
            wavelengths_used.append(wavelength)
            idx_range = (wavelength_positions >= wavelength - search_range) & (wavelength_positions <= wavelength + search_range)
            x_range = wavelength_positions[idx_range]
            y_range = spectrum_data[idx_range]
            
            if len(x_range) == 0 or len(y_range) == 0:
                resolutions.append(None)
                sigmas.append(None)
                continue

            p0 = [np.max(y_range), wavelength, 5, np.min(y_range)]
            try:
                popt, _ = curve_fit(gaussian, x_range, y_range, p0=p0)
                mu, sigma = popt[1], popt[2]
                resolution = mu / (2.355 * sigma)
                resolutions.append(resolution)
                sigmas.append(sigma)
            except RuntimeError:
                resolutions.append(None)
                sigmas.append(None)
        
        resolutions_list.append(resolutions)
        sigmas_list.append(sigmas)
        wavelengths_used_list.append(wavelengths_used)

    # Plot spectral resolution with filtered trendline
    if plotCheck_resolution:
        for i, (wavelengths_used, resolutions) in enumerate(zip(wavelengths_used_list, resolutions_list)):
            plt.figure(figsize=(8, 5))
            valid_points = [
                (w, r) for w, r in zip(wavelengths_used, resolutions)
                if w is not None and r is not None and 4000 <= w <= 6800 and 0 <= r <= 2500
            ]
            if valid_points:
                w_vals, r_vals = zip(*valid_points)
                plt.scatter(w_vals, r_vals, 5, color='black')

                # 2차 추세선 그리기
                if len(w_vals) >= 3:
                    coeffs_trend = np.polyfit(w_vals, r_vals, deg=2)
                    w_fit = np.linspace(4000, 6800, 300)
                    r_fit = np.polyval(coeffs_trend, w_fit)
                    plt.plot(w_fit, r_fit, 'r-')

                plt.xlim(4000, 6800)
                plt.ylim(0, 2500)
                plt.xlabel('Wavelength [Å]')
                plt.ylabel('R=λ/Δλ')
                # plt.title(f'Spectral Resolution for Image {i + 1}')
                # plt.legend()
                # plt.grid()
                plt.show()

    # Plot sigma
    if plotCheck_sigma:
        for i, (wavelengths_used, sigmas) in enumerate(zip(wavelengths_used_list, sigmas_list)):
            plt.figure(figsize=(8, 5))
            valid_points = [(w, s) for w, s in zip(wavelengths_used, sigmas) if w is not None and s is not None]
            if valid_points:
                w_vals, s_vals = zip(*valid_points)
                plt.scatter(w_vals, s_vals, alpha=0.7)
                plt.xlabel('Wavelength [Å]')
                plt.ylabel('σ [Å]')
                # plt.title(f'Gaussian Sigma for Image {i + 1}')
                # plt.grid()
                plt.show()
    
    return resolutions_list, sigmas_list
