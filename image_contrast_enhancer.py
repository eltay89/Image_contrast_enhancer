import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os


def histogram_equalization(img):
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        img_output = cv2.equalizeHist(img)
    return img_output


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_output = clahe.apply(img)
    return img_output


def contrast_stretching(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced_img = enhancer.enhance(1.5)
    img_output = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
    return img_output


def save_image(img, img_path, method):
    filename, ext = os.path.splitext(img_path)
    output_path = f"{filename}_{method}{ext}"
    cv2.imwrite(output_path, img)
    print(f"Enhanced image saved as {output_path}")


def main():
    img_path = input("Enter the path of the image: ")
    img = cv2.imread(img_path)

    if img is None:
        print("Error: Image not found.")
        return

    print("Choose a contrast enhancement technique:")
    print("1. Histogram Equalization")
    print("2. CLAHE")
    print("3. Contrast Stretching")

    choice = int(input("Enter the number of the chosen technique: "))

    if choice == 1:
        enhanced_img = histogram_equalization(img)
        save_image(enhanced_img, img_path, "hist_eq")
    elif choice == 2:
        enhanced_img = clahe(img)
        save_image(enhanced_img, img_path, "clahe")
    elif choice == 3:
        enhanced_img = contrast_stretching(img)
        save_image(enhanced_img, img_path, "cont_stretch")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()