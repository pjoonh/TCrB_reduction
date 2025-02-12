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
            elif 'dark' and '1800s' in uniform_file_name:
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
def extract_spectrum_y_coordi(img_type, new_folder_path, extract_n, extract_dW = 4):
    list_file_path = os.path.join(new_folder_path, f'geo_{img_type}.list')
    
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]

    median_y_data = []
    
    # Process each file path
    for file_path in file_paths:
        # Open the FITS file
        with fits.open(file_path) as hdul:
            img = hdul[0].data  # Get the image data

            # Divide the x-axis into n parts
            x_positions = np.linspace(0, img.shape[1] - 1, extract_n, dtype=int)  # x positions
            peak_y_positions = []

            # Find the peak y-coordinate for each x position
            for x in x_positions:
                peak_region = range(max(0, x - extract_dW), min(img.shape[1], x + extract_dW + 1))
                column_peaks = [np.argmax(img[:, x_val]) for x_val in peak_region]  # Get column values
                peak_y = int(np.median(column_peaks))  # Find peak y-coordinate
                peak_y_positions.append(peak_y)

            # Calculate the median of y-coordinates
            median_y = int(np.median(peak_y_positions))

            # Append the file path and median y-coordinate to the list
            median_y_data.append((file_path, median_y))

    return median_y_data

# 8-2) Obj, Std extract spectrum value
def compute_fwhm(img_type, new_folder_path, median_y_data, sigma=2, n_segments=15):
    list_file_path = os.path.join(new_folder_path, f'geo_{img_type}.list')
    
    with open(list_file_path, 'r') as list_file:
        file_paths = [line.strip() for line in list_file if line.strip()]

    fwhm_data = []
    
    for file_path, median_y in median_y_data:
        with fits.open(file_path) as hdul:
            img = hdul[0].data  # Get the image data
            fwhm_values = []
            
            for x in range(img.shape[1]):  # Iterate through each column
                column_data = img[:, x]  # Extract column data
                smoothed_column = gaussian_filter1d(column_data, sigma=sigma)  # Apply Gaussian smoothing
                
                peak_idx, _ = find_peaks(smoothed_column)  # Find peaks
                if len(peak_idx) == 0:
                    fwhm_values.append(None)
                    continue
                
                peak_y = peak_idx[np.argmax(smoothed_column[peak_idx])]  # Select highest peak
                half_max = smoothed_column[peak_y] / 2  # Half max value
                
                left_idx = np.where(smoothed_column[:peak_y] < half_max)[0]
                right_idx = np.where(smoothed_column[peak_y:] < half_max)[0] + peak_y
                
                if len(left_idx) == 0 or len(right_idx) == 0:
                    fwhm_values.append(None)
                    continue
                
                left = left_idx[-1]
                right = right_idx[0]
                fwhm = right - left
                
                fwhm_values.append(fwhm)
            
            # Split into segments and compute median FWHM for each segment
            segment_size = len(fwhm_values) // n_segments
            segmented_fwhm = []
            for i in range(n_segments):
                start = i * segment_size
                end = start + segment_size if i < n_segments - 1 else len(fwhm_values)
                segment_fwhm = [f for f in fwhm_values[start:end] if f is not None]
                median_fwhm = np.median(segment_fwhm) if segment_fwhm else None
                segmented_fwhm.append(median_fwhm)
            
            fwhm_data.append((file_path, segmented_fwhm))
    
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
                
                lower_inner = int(median_y - (fwhm_inner / 2) * fwhm_rep)
                lower_outer = int(median_y - fwhm_outer * fwhm_rep)
                upper_inner = int(median_y + (fwhm_inner / 2) * fwhm_rep)
                upper_outer = int(median_y + fwhm_outer * fwhm_rep)
                
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
def compute_spectrum(img_type, new_folder_path, median_y_data, fwhm_data, background_values, fwhm_spectrum=4, plotCheck=False):
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
                fwhm_rep = segmented_fwhm[segment_idx]

                if fwhm_rep is None or background_column_values[x] is None:
                    raw_spectrum_values.append(None)
                    corrected_spectrum_values.append(None)
                    continue

                # 스펙트럼 구간 설정
                lower_bound = int(median_y - (fwhm_spectrum / 2) * fwhm_rep)
                upper_bound = int(median_y + (fwhm_spectrum / 2) * fwhm_rep)

                # 이미지 경계를 벗어나지 않도록 조정
                lower_bound = max(0, lower_bound)
                upper_bound = min(img.shape[0], upper_bound)

                # 해당 열에서 스펙트럼 값 계산 (구간 내 데이터 합)
                spectrum_value = np.sum(img[lower_bound:upper_bound, x])

                # 배경 보정: 배경값 * fwhm_spectrum 길이만큼 빼줌
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

    return spectrum_values


# 임의로 양쪽에서 일부 데이터 지움 - 2픽셀 정도
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
                    
                    lower_bound = int(median_y - (fwhm_arc_spectrum / 2) * fwhm_rep)
                    upper_bound = int(median_y + (fwhm_arc_spectrum / 2) * fwhm_rep)
                    
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
def wavelength_equation(img_type, new_folder_path, sigma=2):
    # 폴더 내의 extrac_arc{숫자}_{img_type}.list 파일 검색
    list_files = [f for f in os.listdir(new_folder_path) if f.startswith(f'extrac_arc') and f.endswith(f'_{img_type}.list')]

    for list_file in list_files:
        # print(f"\nProcessing {list_file}...")
        
        # .list 파일에서 arc 번호 추출
        arc_number = list_file.split('_')[1]  # arc1, arc2 등

        # .list 파일 열기
        with open(os.path.join(new_folder_path, list_file), 'r') as file:
            arc_files = file.readlines()

        # 각 아크 파일에 대해 반복 처리
        for arc_file in arc_files:
            arc_file = arc_file.strip()
            # print(f"  - Processing {arc_file}...")

            # FITS 파일 열기
            with fits.open(os.path.join(new_folder_path, arc_file)) as hdul:
                data = hdul[0].data  # 첫 번째 HDU에서 데이터 가져오기

            # 가우시안 스무딩 적용
            smoothed_data = gaussian_filter1d(data, sigma=sigma)

            # 모든 피크 찾기
            peaks, _ = find_peaks(smoothed_data)

            # 피크들의 높이 계산
            peak_heights = smoothed_data[peaks]

            # 피크의 높이 순으로 상위 3개 피크 선택
            top_3_indices = np.argsort(peak_heights)[-3:]  # 가장 높은 3개 피크의 인덱스
            top_3_peaks = peaks[top_3_indices]  # 실제 피크 위치
            top_3_heights = peak_heights[top_3_indices]  # 해당 피크의 값

            # x인덱스 값을 작은 것부터 정렬
            sorted_top_3_peaks = np.sort(top_3_peaks)

            # 3개의 피크에 해당하는 파장 값
            peak_wavelengths = [5852.49, 6143.06, 6402.25]

            # 선형 회귀를 사용하여 x인덱스와 파장 매칭
            coeffs = np.polyfit(sorted_top_3_peaks, peak_wavelengths, deg=1)  # 선형 회귀

            # 회귀 계수
            a, b = coeffs

            # 함수식 출력
            print(f"    - 계산된 함수식: y = {a:.5f}x + {b:.5f}")

            # 계산된 선형 방정식
            # known_neon_wavelengths 배열
            known_neon_wavelengths = np.array([
                5037.37, 5330.78, 5341.09, 5400.58, 5719.23, 5748.30, 5764.42, 5804.45, 5820.16,
                5852.49, 5881.89, 5944.83, 5975.53, 6030.00, 6074.34, 6096.16, 6143.06, 6163.59, 6217.28, 
                6266.49, 6304.79, 6334.43, 6382.99, 6402.25, 6506.53, 6532.88, 6598.95, 6678.28, 6717.04
            ])

            # 시각화
            # plt.figure(figsize=(10, 5))
            # plt.plot(data, label="Original Data", alpha=0.5)
            # plt.plot(smoothed_data, label="Smoothed Data", linewidth=2)

            # 3개의 피크에 해당하는 파장 표시 (세로선과 파장 숫자)
            for i, peak in enumerate(sorted_top_3_peaks):
                wavelength = peak_wavelengths[i]  # 원래 파장 값
                plt.axvline(x=peak, color='red', linestyle='--', label=f"Peak {i+1} at {wavelength} Å")
                plt.text(peak, smoothed_data[peak], f"{wavelength} Å", color='red', fontsize=12, ha='center', va='bottom')

            # 계산된 선형 회귀에 의한 다른 파장 값에 대한 x 인덱스
            for wavelength in known_neon_wavelengths:
                x_index = (wavelength - b) / a  # x = (y - b) / a
                plt.axvline(x=x_index, color='blue', linestyle=':')
                # plt.text(x_index, np.max(smoothed_data)*0.9, f"{wavelength} Å", color='blue', fontsize=10, ha='center', va='bottom')

            # plt.xlabel("Index")
            # plt.ylabel("Intensity")
            # plt.title(f"Arc {arc_number} - {arc_file} Spectrum with Peaks and Known Neon Wavelengths")
            # plt.legend()
            # plt.show()

            # x인덱스와 파장 매핑 출력
            # print("    - X Index to Wavelength Mapping:")
            # for index, wavelength in zip(sorted_top_3_peaks, peak_wavelengths):
                # print(f"      Index: {index}, Wavelength: {wavelength} Å")

            # 계산된 x 인덱스와 파장 출력
            # for wavelength in known_neon_wavelengths:
            #     x_index = (wavelength - b) / a
            #     print(f"      Wavelength: {wavelength} Å -> Index: {x_index}")

        print(f"Completed processing {list_file}.\n")