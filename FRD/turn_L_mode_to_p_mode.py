import os
from PIL import Image
from tqdm import tqdm
import numpy as np

def gen_cmap(dataset='voc', normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    if dataset == 'voc':
        N = 256
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])
    elif dataset == 'postdam':
        N = 6
        cmap = np.zeros((N, 3), dtype=dtype)
        cmap[0] = np.array([0, 0, 0])        # red  (background)
        cmap[1] = np.array([255, 255, 255])  # white  (Impervious surfaces)
        cmap[2] = np.array([0, 0, 255])      # Blue   (Building)
        cmap[3] = np.array([0, 255, 255])    # Cyan   (Low vegetation)
        cmap[4] = np.array([0, 255, 0])      # Green  (Tree)
        cmap[5] = np.array([255, 255, 0])    # Yellow (Car)

    cmap = cmap/255 if normalized else cmap
    return cmap

def colorize_image_postdam(image, dataset):
    """根據指定的顏色值塗色圖片"""
    color_map = gen_cmap(dataset)
    
    colored_image = Image.new("P", image.size)
    palette = []
    for i in range(256):  # 初始化調色板，默認為黑色
        palette.extend((0, 0, 0))
    for key, val in enumerate(color_map):
        palette[key*3:key*3+3] = val.astype(int).tolist()
    colored_image.putpalette(palette)

    pixels = colored_image.load()

    for i in range(image.size[0]):
        for j in range(image.size[1]):
            color_index = image.getpixel((i, j))
            if color_index < len(color_map):
                pixels[i, j] = color_index
            else:
                pixels[i, j] = 0  # 預設為黑色

    return colored_image
# MCTformer_offical
# /result_dir/MCTformer_results/mct_prototypes_b128_th49_step/pseudo-mask-ms
image_path = "./result_dir/tsai/MCTformer_results/nslp/pseudo-mask-ms-crf"
output_path = image_path + "-colorized"
data_set = 'postdam' # 'voc' or 'postdam' postdam適用於postdam、vaihingen等數據集
image_list = os.listdir(image_path)
image_list = [f for f in image_list if f.endswith('.png')]
os.makedirs(output_path, exist_ok=True)
print(f'image path: {image_path}\noutput path: {output_path}')
for image_name in tqdm(image_list, desc='Colorizing images'):
    image = Image.open(os.path.join(image_path, image_name))
    colored_image = colorize_image_postdam(image, data_set)
    colored_image.save(os.path.join(output_path, image_name))