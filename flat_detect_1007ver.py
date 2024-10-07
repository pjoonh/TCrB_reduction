# ==============================flat 검출 10.07ver(이미지 자르기 추가)=============================
'''
1. 데이터 뽑기 (2D) - l_dat, r_dat
2. median으로 한 줄로 만들기 - l_med, r_med
3. detect edge
 1) 가우시안 필터 스무딩 (전후비교 show) - l_gau, r_gau
 2) 기울기가 기준 이상인 점 다 뽑기(인덱스) - l_edges, r_edges(l1_y = l_edges[0], ...)
4. gau data와 edges를 show
5. 좌표 설정 및 직선 뽑기
 1) 4개의 점을 위한 x좌표 설정 - l_x, r_X 
 2) 직선 뽑기 - line1, line2
6. 이미지 잘라서 fits로 저장
'''
# 중간중간 데이터 확인용으로 작성한 부분은 주석처리 해놓음

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 이미지 불러오기
fits_image_path = 'Flat-001.fit'
hdu = fits.open(fits_image_path)[0]
dat = hdu.data

##################################################################################################
# 1. 데이터 뽑기 (2D) - l_dat, r_dat
dW = 10
shift = 1500
height, width = dat.shape
l_dat = dat[:, shift : shift+dW+1]
r_dat = dat[:, width-dW-1 : width]
##################################################################################################
# 2. median으로 한 줄로 만들기 - l_med, r_med
l_med = np.median(l_dat, axis=1)
r_med = np.median(r_dat, axis=1)

################################################################### 데이터 확인용
# print(f"Median values: {r_med}")
# print(f"Median indices: {r_med_indices}")
###################################################################

##################################################################################################
# 3-1. 가우시안 필터 스무딩 (전후비교 show) - l_gau, r_gau
sigma = 2
l_gau = gaussian_filter1d(l_med, sigma=sigma)
r_gau = gaussian_filter1d(r_med, sigma=sigma)

# 가우시안 전후를 비교하여 보여주기 위한 함수
# def median_n_gaussian(l_med, l_gau, r_med, r_gau):
#     # 1x2 subplot 생성
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
#     # 첫 번째 subplot에 l_med와 l_gau를 그림
#     axs[0].plot(l_med, label='l_med')
#     axs[0].plot(l_gau, label='l_gau')
#     axs[0].set_title('l_med vs l_gau')
#     axs[0].legend()

#     # 두 번째 subplot에 r_med와 r_gau를 그림
#     axs[1].plot(r_med, label='r_med')
#     axs[1].plot(r_gau, label='r_gau')
#     axs[1].set_title('r_med vs r_gau')
#     axs[1].legend()

#     # 레이아웃을 조정하여 플롯 간의 간격을 맞춤
#     plt.tight_layout()

#     # 플롯을 화면에 출력
#     plt.show()

# median_n_gaussian(l_med, l_gau, r_med, r_gau)

# 3-2. 기울기가 기준 이상인 점 다 뽑기(인덱스) - l_edges, r_edges(l1_y = l_edges[0], ...)
def detect_edges(arr, threshold):    
    # 1차 미분 계산
    diff = np.diff(arr)
    
    # 차이가 threshold 이상인 위치 찾기
    edge_indices = np.where(np.abs(diff) > threshold)[0]
    
    return edge_indices.tolist()

l_edges = detect_edges(l_gau, threshold=25)
r_edges = detect_edges(r_gau, threshold=40)

#################################################################################### 데이터 확인용
# print(l_edges)
# print(r_edges)
####################################################################################

# 엣지의 y값 인덱스 저장
l1_y = l_edges[0]
l2_y = l_edges[-1]
r1_y = r_edges[0]
r2_y = r_edges[-1]

##################################################################################################
# 4. gau data와 edges를 show
# def plot_edges(l_gau, r_gau, l1_y, l2_y, r1_y, r2_y):
#     # 1x2 서브플롯을 생성하여 옆으로 배치
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
#     # 첫 번째 subplot에 l_gau와 두 상수값에 해당하는 세로 점선을 그림
#     axs[0].plot(l_gau, label='l_gau')
#     axs[0].axvline(x=l1_y, color='r', linestyle='--', label='l1_y')
#     axs[0].axvline(x=l2_y, color='b', linestyle='--', label='l2_y')
#     axs[0].set_title('l_gau with l1_y and l2_y')
#     axs[0].legend()

#     # 두 번째 subplot에 r_gau와 두 상수값에 해당하는 세로 점선을 그림
#     axs[1].plot(r_gau, label='r_gau')
#     axs[1].axvline(x=r1_y, color='r', linestyle='--', label='r1_y')
#     axs[1].axvline(x=r2_y, color='b', linestyle='--', label='r2_y')
#     axs[1].set_title('r_gau with r1_y and r2_y')
#     axs[1].legend()

#     # 레이아웃을 조정하여 플롯 간의 간격을 맞춤
#     plt.tight_layout()

#     # 플롯을 화면에 출력
#     plt.show()

# plot_edges(l_gau, r_gau, l1_y, l2_y, r1_y, r2_y)

##################################################################################################
# 5-1. 4개의 점을 위한 x좌표 설정 - l_x, r_X 
l_x = shift + dW/2
r_x = width - dW/2

# 5-2. 직선 뽑기 - line1, line2
def line_calc(l_x, l_y, r_x, r_y):
    slope = (r_y - l_y) / (r_x - l_x)
    intercept = l_y - slope * l_x
    return slope, intercept

slope1, intercept1 = line_calc(l_x, l1_y, r_x, r1_y)
slope2, intercept2 = line_calc(l_x, l2_y, r_x, r2_y)

##################################################################데이터 확인용
# 점과 직선 plot
# plt.imshow(dat, cmap='gray')
# plt.plot(r_x, r1_y, 'o', color = 'white', ms=5)
# plt.plot(r_x, r2_y, 'o', color = 'white', ms=5)
# plt.plot(l_x, l1_y, 'o', color = 'white', ms=5)
# plt.plot(l_x, l2_y, 'o', color = 'white', ms=5)

# x = np.arange(0, width)
# plt.plot(x, slope1 * x + intercept1, color='red', label='line1')
# plt.plot(x, slope2 * x + intercept2, color='red', label='line2')

# plt.legend()
# plt.show()
##################################################################



##################################################################################################
# 6. 이미지 잘라서 fits로 저장
# (dat을 NaN을 지원하는 형식인 float로 바꿔서 NaN으로 저장해도 0으로 저장됨을 확인함, dat=dat.astype(float))
# (png파일의 투명화 기능처럼 바깥 부분을 아예 지우려고 했지만, 행의 길이가 달라서 코드를 실행하지 못한다는 메세지가 뜸)

# 출력 파일 경로
output_fits_file = 'detected_Flat-001.fits'

# 필요한 데이터에서 이미지를 자르기 위한 y값을 추출
def calculate_y(height, width, slope1, intercept1, slope2, intercept2):
    # line1과 line2의 x좌표를 0과 width로 설정하여 y좌표 계산
    y1_start = slope1 * 0 + intercept1  # x = 0
    y1_end = slope1 * width + intercept1  # x = width
    
    y2_start = slope2 * 0 + intercept2  # x = 0
    y2_end = slope2 * width + intercept2  # x = width

    # y좌표 중에서 최대값과 최소값을 계산
    y_min = max([y1_start, y1_end])
    y_max = min([y2_start, y2_end])

    return y_max, y_min

y_max, y_min = calculate_y(height, width, slope1, intercept1, slope2, intercept2)

##################################################################데이터 확인용
# print(f'M : {y_max} and m : {y_min}')
##################################################################

# 이미지 자르기
Min = int(np.ceil(y_min))
Max = int(np.trunc(y_max))

cropped_dat = dat[Min:Max, :] # [y좌표, x좌표표]

##################################################################데이터 확인용
# print(f'{Max} And {Min}')
# plt.imshow(cropped_dat, cmap='gray')
# plt.show()
##################################################################

# 두 직선 사이의 영역만 유지, 나머지는 0으로
modified_dat = np.zeros_like(cropped_dat)

# 새로운 이미지에 맞춰 값 조정
new_height, new_width = cropped_dat.shape # 이미지 크기 변동
new_intercept1 = intercept1 - Min # line1의 y절편 변동
new_intercept2 = intercept2 - Min # line2의 y절편 변동

# 이미지의 각 픽셀을 판별(직선 안에 있는지)
for y in range(new_height):
    for x in range(new_width):
        # 각 직선 위의 y 좌표를 계산
        line1 = slope1 * x + new_intercept1
        line2 = slope2 * x + new_intercept2
            
        # 해당 픽셀이 두 직선 사이에 있는지 확인
        if min(line1, line2) <= y <= max(line1, line2):
            modified_dat[y, x] = cropped_dat[y, x]  # 두 직선 사이에 있는 데이터만 유지

# 새로운 FITS 파일 저장
hdu = fits.PrimaryHDU(np.array(modified_dat))
hdul_new = fits.HDUList([hdu])
hdul_new.writeto(output_fits_file, overwrite=True)
    
print(f"Saved to {output_fits_file}.")
