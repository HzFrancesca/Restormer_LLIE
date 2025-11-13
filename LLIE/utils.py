import os

import cv2


def load_img(path: str):
    """读取图像并返回 RGB numpy 数组."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_img(path: str, img):
    """将 RGB numpy 数组保存到指定路径."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)