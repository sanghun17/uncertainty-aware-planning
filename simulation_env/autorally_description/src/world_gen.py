import os
import numpy as np
from PIL import Image

version = 2
map_png_dir = '/home/mlpc/offroad_ws/src/simulation_env/autorally_description/urdf/map_png'
if not os.path.exists(map_png_dir):
    os.makedirs(map_png_dir)
save_path = os.path.join(map_png_dir, f'height_map_v{version}.png')  


# 이미지 크기와 sin 함수의 주기, 진폭, 위상 등을 설정
width = 65
height = 65
period = 10
amplitude = 0.1
phase = 0

y, x = np.indices((height, width))
data = amplitude * np.sin(2 * np.pi * (1.0 / period) * x + phase)

img = Image.fromarray(np.uint8(data * 255))


img.save(save_path)
print("done")