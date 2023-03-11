# Author: OedoSoldier [大江户战士]
# https://space.bilibili.com/55123

from PIL import Image, ImageFilter
import numpy as np


class CropUtils(object):
    """
    This class provides utility functions for cropping and restoring images.

    The `crop_img()` function takes an image and a corresponding mask, and uses the mask to
    crop the image to the minimum bounding box that includes the non-zero pixels in the mask.
    If the width and height of the resulting image are not equal, the image is scaled up to a
    square image using zero padding. The function returns the cropped image, the cropped mask,
    and the bounding box and image size as a tuple.

    The `restore_by_file()` function takes a raw image, a cropped image, a reference image,
    and a blur mask, and uses these images to restore the cropped image to the raw image.
    The reference image is used to determine the bounding box of the cropped image, and the
    blur mask is used to apply a gaussian blur to the alpha channel of the cropped image.
    The function returns the restored image.
    """

    def crop_img(self, img, mask, threshold=50):
        """
        Crop the given image using the given mask.

        Args:
            img: The image to be cropped, as a PIL.Image object.
            mask: The mask to be used for cropping, as a PIL.Image object.
            threshold: The threshold to use for converting the mask to binary. Pixels in the mask
                       with a value greater than the threshold will be considered as part of the
                       mask, and will be included in the cropped image. Pixels with a value less
                       than or equal to the threshold will be ignored. (default: 50)

        Returns:
            A tuple containing the cropped image, the cropped mask, and a tuple with the bounding
            box and image size. If the mask is empty, the function returns (img, None, None).
        """

        # Code for cropping the image using the mask

        mask = mask.resize(img.size) if img.size[0] != mask.size[0] else mask

        bbox = mask.convert('L').point(
            lambda x: 255 if x > threshold else 0,
            mode='1').getbbox()

        if bbox:
            img, mask = img.crop(bbox), mask.crop(bbox)
            size = img.size
            if size[0] != size[1]:
                bigside = size[0] if size[0] > size[1] else size[1]

                img_np = np.zeros((bigside, bigside, 4), dtype=np.uint8)
                mask_np = np.zeros((bigside, bigside, 4), dtype=np.uint8)

                offset = (
                    round(
                        (bigside - size[0]) / 2),
                    round(
                        (bigside - size[1]) / 2))

                img_np[offset[1]:offset[1] + size[1],
                       offset[0]:offset[0] + size[0]] = img
                mask_np[offset[1]:offset[1] + size[1],
                        offset[0]:offset[0] + size[0]] = mask

                img = Image.fromarray(img_np)
                mask = Image.fromarray(mask_np)

            return img, mask, bbox + size

        return img, None, None

    def restore_by_file(
            self,
            raw,
            img,
            ref_img,
            blur_mask,
            info,
            mask_blur=0.5):
        """
        Restore the given cropped image to the given raw image.

        Args:
            raw: The raw image, as a PIL.Image object.
            img: The cropped image, as a PIL.Image object.
            ref_img: The reference image, as a PIL.Image object. This image is used to determine
                     the bounding box of the cropped image.
            blur_mask: The blur mask, as a PIL.Image object. This mask is used to apply a gaussian
                       blur to the alpha channel of the cropped image.
            info: A tuple containing the bounding box of the cropped image. This tuple should have
                  the form (upper_left_x, upper_left_y, lower_right_x, lower_right_y).
            mask_blur: The sigma value to use for the gaussian blur. Higher values result in a
                       stronger blur. (default: 0.5)

        Returns:
            The restored image, as a PIL.Image object.
        """

        # Code for restoring the cropped image

        raw_size = raw.size
        ref_size = ref_img.size

        upper_left_x = info[0]
        upper_left_y = info[1]

        img = img.resize(ref_size).convert('RGBA')
        blur_mask = blur_mask.resize(ref_size).convert('RGBA')
        raw = raw.convert('RGBA')

        bbox = ref_img.split(
        )[-1].convert('L').point(lambda x: 255 if x > 0 else 0, mode='1').getbbox()
        bbox = list(bbox)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        img = img.crop(bbox)
        blur_mask = blur_mask.crop(bbox)

        blur_img = np.zeros((raw_size[1], raw_size[0], 4), dtype=np.uint8)
        blur_img[upper_left_y:upper_left_y +
                 h, upper_left_x:upper_left_x +
                 w, :] = np.array(blur_mask)
        blur_img = Image.fromarray(blur_img, 'RGBA')
        blur_img = blur_img.filter(ImageFilter.GaussianBlur(mask_blur))

        new_img = np.zeros((raw_size[1], raw_size[0], 4), dtype=np.uint8)
        new_img[upper_left_y:upper_left_y +
                h, upper_left_x:upper_left_x +
                w, :] = np.array(img)
        new_img = Image.fromarray(new_img, 'RGBA')

        new_img = Image.alpha_composite(raw, new_img)
        new_img.putalpha(blur_img.split()[-1].convert('L'))
        new_img = Image.alpha_composite(raw, new_img)

        return new_img
