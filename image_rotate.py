##################################################################################################
## 초기 설정 : 16번줄(이미지 경로), 21번줄(데이터 열을 자를 너비), 86번줄(회전된 이미지 저장경로) ##
##################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from scipy.ndimage import rotate

# 1. 가우시안 함수 정의
def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset

# 2. FITS 이미지 파일 불러오기
fits_image_path = 'PiSer-001.fit'  # 이미지 경로 입력
hdu = fits.open(fits_image_path)[0]
spectrum_image = hdu.data

# 3. 세로 n픽셀 간격으로 중앙값 취해서 데이터 만들기  를 위한 세팅
n = 50  # 몇 픽셀 간격으로 데이터 자를 건지 설정
height, width = spectrum_image.shape  # 이미지의 크기 저장
column_indices = np.arange(0, width, n)  # n픽셀 간격으로 열 선택(0에서 width까지 n간격으로)
median_values = np.zeros(len(column_indices))  # 중앙값 저장할 array 만듦

# 4. 가우시안 피팅 후 피크값 추출 (chatgpt4o)
peak_positions = []
x_data = np.arange(height)  # 마스킹된 이미지 내의 좌표

for i, col_idx in enumerate(column_indices):
    # 각 열의 데이터
    y_data = spectrum_image[:, col_idx]

    # 픽셀값이 너무 낮은 데이터는 건너 뜀(노이즈 1차 제거)
    if np.max(y_data) < 0.1 * np.max(spectrum_image):
        peak_positions.append(np.nan)
        continue

    # 가우시안 초기값 설정
    amp_guess = np.max(y_data) - np.min(y_data)  # 신호의 범위
    mu_guess = np.argmax(y_data)  # 가장 큰 값의 위치
    sigma_guess = 10  # 표준편차
    offset_guess = np.min(y_data)  # 최소값을 오프셋으로 설정
    initial_guess = [amp_guess, mu_guess, sigma_guess, offset_guess]

    # curve fitting
    try:
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess, maxfev=10000)
        # 피크 위치는 마스킹한 이미지 기준이므로 원본 이미지 기준으로 보정
        peak_positions.append(popt[1])
    except RuntimeError:
        # 피팅 실패 시 이전 열의 피크값으로 대체
        if i > 0:
            peak_positions.append(peak_positions[-1])
        else:
            peak_positions.append(mu_guess)  # 첫 번째 열이면 초기 추정값 사용

# 5. 원본 2D 이미지 띄우기
plt.imshow(spectrum_image, cmap='gray', origin='lower')

# 6. 간격 n으로 추출된 열에 해당하는 피크값을 선으로 연결
plt.plot(column_indices, peak_positions, color='red', linestyle='-', linewidth=0.5)

plt.title('Spectrum Image with Fitted Gaussian Peaks')
plt.show()

# 7. nan 값이 없는 데이터만 추출
valid_indices = ~np.isnan(peak_positions)  # nan이 아닌 값들의 인덱스
valid_column_indices = column_indices[valid_indices]
valid_peak_positions = np.array(peak_positions)[valid_indices]

# 8. 선형 추세선 구하기
slope, intercept = np.polyfit(valid_column_indices, valid_peak_positions, 1)  # 1차 다항식에 피팅

# 9. 1차식 출력
print(f"Linear Trendline: y = {slope:.4f}x + {intercept:.4f}")

# 10. slope로 회전 각도 계산
rotation_angle = np.degrees(np.arctan(slope))
# print(f'rotation angle : {rotation_angle}')

# 11. 이미지를 회전 (rotate 함수 사용, 빈 공간은 0으로 채움)
rotated_image = rotate(spectrum_image, rotation_angle, reshape=False, cval=0)

# 12. 회전된 이미지를 FITS 파일로 저장
rotated_fits_path = 'save_path'  # 저장할 경로
hdu_rotated = fits.PrimaryHDU(rotated_image)
hdu_rotated.writeto(rotated_fits_path, overwrite=True)

print(f"Saved to {rotated_fits_path}")
