"""
Image preprocessing module: supports aspect-ratio-preserving resize and vertical text rotation
"""
import torch
from PIL import Image
import torchvision.transforms.functional as F


def is_vertical_image(image):
    """
    Determine if image contains vertical text
    Based on aspect ratio: height > width indicates vertical text
    """
    width, height = image.size
    return height > width


def rotate_vertical_image(image):
    """
    Rotate vertical text image 90 degrees counterclockwise
    PIL.Image.rotate() uses counterclockwise as positive direction
    """
    return image.rotate(90, expand=True)


def resize_with_padding(image, target_size=(384, 384), fill_color=(255, 255, 255)):
    """
    Resize while preserving aspect ratio, pad with white borders

    Args:
        image: PIL Image
        target_size: (width, height) target size
        fill_color: Fill color, default white (255, 255, 255)

    Returns:
        PIL Image: Processed image
    """
    target_width, target_height = target_size
    original_width, original_height = image.size

    # Calculate scaling ratio (preserve aspect ratio, take minimum to ensure image doesn't exceed target size)
    ratio_w = target_width / original_width
    ratio_h = target_height / original_height
    ratio = min(ratio_w, ratio_h)

    # Calculate resized dimensions
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize image (preserve aspect ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create white background with target size
    new_image = Image.new("RGB", target_size, fill_color)

    # Calculate paste position (center)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste resized image onto white background
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def preprocess_image_for_ocr(image, target_size=(384, 384), auto_rotate_vertical=True):
    """
    Complete OCR image preprocessing pipeline

    Args:
        image: PIL Image (RGB)
        target_size: (width, height) target size
        auto_rotate_vertical: Whether to auto-rotate vertical text

    Returns:
        PIL Image: Processed image
    """
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Check if vertical text, rotate 90 degrees counterclockwise if so
    if auto_rotate_vertical and is_vertical_image(image):
        image = rotate_vertical_image(image)

    # Resize with aspect ratio preservation + padding
    image = resize_with_padding(image, target_size=target_size)

    return image


def image_to_tensor(image, normalize=True):
    """
    Convert PIL Image to Tensor

    Args:
        image: PIL Image
        normalize: Whether to normalize (mean=0.5, std=0.5)

    Returns:
        torch.Tensor: shape (3, H, W)
    """
    # Convert to Tensor
    tensor = F.to_tensor(image)

    # Normalize
    if normalize:
        tensor = F.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    return tensor


class OCRImageTransform:
    """
    OCR image transform class (can be used in Dataset)
    """
    def __init__(self, target_size=(384, 384), auto_rotate_vertical=True, normalize=True):
        self.target_size = target_size
        self.auto_rotate_vertical = auto_rotate_vertical
        self.normalize = normalize

    def __call__(self, image):
        """
        Args:
            image: PIL Image

        Returns:
            torch.Tensor: shape (3, H, W)
        """
        # Preprocess image
        image = preprocess_image_for_ocr(
            image,
            target_size=self.target_size,
            auto_rotate_vertical=self.auto_rotate_vertical
        )

        # Convert to Tensor
        tensor = image_to_tensor(image, normalize=self.normalize)

        return tensor
