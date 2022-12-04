# Author: OedoSoldier [大江户战士]
# https://space.bilibili.com/55123

from PIL import Image, ImageOps, ImageFilter
import numpy as np


class CropUtils(object):
    def crop_img(self, img, mask, threshold=50):
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
