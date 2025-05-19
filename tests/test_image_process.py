import os, sys
from PIL import Image

# 获取 lite_llama 目录的绝对路径并添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lite_llama.lite_llama.utils.image_process import vis_images


def test_vis_images(image_files):
    print("=" * 50)
    print("Input Image:")
    vis_images(image_files)


if __name__ == "__main__":
    image_files = [
        "/gemini/code/lite_llama/images/pexels-christian-heitz-285904-842711.jpg",
        "/gemini/code/lite_llama/images/pexels-francesco-ungaro-1525041.jpg",
    ]
    test_vis_images(image_files)
