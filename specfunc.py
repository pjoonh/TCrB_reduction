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
    write_to_list_file(new_folder_path, 'bias.list', bias_files)
    write_to_list_file(new_folder_path, 'dark_300s.list', dark_300s_files)
    write_to_list_file(new_folder_path, 'dark_1800s.list', dark_1800s_files)
    write_to_list_file(new_folder_path, 'piser.list', piser_files)
    write_to_list_file(new_folder_path, 'tcrb.list', tcrb_files)

    # Now handle arc and flat files with directory traversal
    arc_folder, flat_folder = find_arc_and_flat_files(parent_folder, arc_files, flat_files)

    # Write arc and flat files to .list files
    write_to_list_file(new_folder_path, 'arc.list', arc_files)
    write_to_list_file(new_folder_path, 'flat.list', flat_files)

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

def write_to_list_file(new_folder_path, file_name, files):
    # Write the sorted file names to the .list file in the new folder path
    list_file_path = os.path.join(new_folder_path, file_name)
    if files:  # If there are files in the list
        with open(list_file_path, 'w') as f:
            for file in files:
                f.write(file + '\n')
        print(f"Written {len(files)} files to {list_file_path}")
    else:
        print(f"No files found for {file_name}, skipping write.")


# 2. edge detection
# 2-1) edge1_smoothing : 
def edge1_smoothing(new_folder_path, dW = dW, shift = shift, sigma = sigma, GV = False):
    # Select path for first image among flat images
    flat_image_path = path_of_the_first_flat(f'new_folder_path + "/flat.list"')
    hdu = fits.open(flat_image_path)[0]
    dat = hdu.data
    
    # Extract data of dW interval from both sides of image
    height, width = dat.shape
    l_dat = dat[:, shift : (shift + dW + 1)]
    r_dat = dat[:, (width - dW - 1) : width]
     
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

def path_of_the_first_flat(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.readline().strip() # Remove '\n' after reading the first line of the file
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
def edge2_detection(l_gau, r_gau, l_thrsh = l_thrsh, r_thrsh = r_thrsh, EV = False):
    # Extract the first point where the slope changes sharply on the left and right
    l_edges = detect_edges(l_gau, l_thrsh)
    r_edges = detect_edges(r_gau, r_thrsh)
    
    # y-coordinate of edge
    l1_y = l_edges[0]
    l2_y = l_edges[-1]
    r1_y = r_edges[0]
    r2_y = r_edges[-1]
    
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
def edge3_minmax(l1_y, l2_y, r1_y, r2_y, dW = dW, shift = shift):
    # x-coordinate of edge
    l_x = shift + dW/2
    r_x = width - dW/2
    
    # Calculate the equation of a straight line
    slope1, intercept1 = line_calc(l_x, l1_y, r_x, r1_y)
    slope2, intercept2 = line_calc(l_x, l2_y, r_x, r2_y)
    
    # Top and bottom values ​​of edge y coordinate
    y_max, y_min = edge_y_minmax(slope1, intercept1, slope2, intercept2)
    
    return y_max, y_min

def line_calc(l_x, l_y, r_x, r_y):
    slope = (r_y - l_y) / (r_x - l_x)
    intercept = l_y - slope * l_x
    return slope, intercept

def edge_y_minmax(slope1, intercept1, slope2, intercept2):
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
                cropped_img = img[Min:Max, :]  # Crop based on input

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
    write_to_list_file(new_folder_path, f'{process}_bias.list', bias_files)
    write_to_list_file(new_folder_path, f'{process}_dark_300s.list', dark_300s_files)
    write_to_list_file(new_folder_path, f'{process}_dark_1800s.list', dark_1800s_files)
    write_to_list_file(new_folder_path, f'{process}_piser.list', piser_files)
    write_to_list_file(new_folder_path, f'{process}_tcrb.list', tcrb_files)
    write_to_list_file(new_folder_path, f'{process}_arc.list', arc_files)
    write_to_list_file(new_folder_path, f'{process}_flat.list', flat_files)    

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
def cr_removal(img_type, new_folder_path, sigclip = sigclip, sigfrac = sigfrac, gain = gain, readnoise = readnoise):
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
# 7-1) rotation angle
def geometry_twice(new_folder_path):
    rotation_angle1 = geometry_angle(new_folder_path, reps = 0)
    rotation_angle2 = geometry_angle(new_folder_path, reps = 1)
    total_angle = rotation_angle1 + rotation_angle2
    
    return total_angle

def geometry_angle(new_folder_path, reps):
    # Select path for first image among flat images
    if reps == 0:
        flat_image_path = os.path.join(new_folder_path, f'crr_flat.list')
    elif reps == 1:
        flat_image = os.path.join(new_folder_path, f'rot_flat.fits')
        
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
    
    rotated_fits_path = os.path.join(new_folder_path, f'rot_flat.fits')
    hdu_rotated = fits.PrimaryHDU(rotated_image)
    hdu_rotated.writeto(rotated_fits_path, overwrite = True)
    
    return rotation_angle

def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset

# 7-2) rotation all images
def geometry_img(img_type, new_folder_path, rotation_angle):
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
                geometry_img = rotate(img, rotation_angle, reshape = False, cval = 0)

                # Append the file path and rotated data to the list
                geo_data_list.append((file_path, geometry_img))

    return geo_data_list
    