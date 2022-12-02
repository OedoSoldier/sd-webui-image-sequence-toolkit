# Author: OedoSoldier [大江户战士]
# https://space.bilibili.com/55123

from PIL import Image, ImageOps, ImageFilter


class CropUtils(object):
    def crop_img(self, img, mask, threshold=50):
        if img.size[0] != mask.size[0]:
            mask = mask.resize(img.size)
        bbox = mask.convert('L').point(
            lambda x: 255 if x > threshold else 0,
            mode='1').getbbox()
        if not bbox:
            return img, None, None
        bbox = list(bbox)
        info = bbox
        img = img.crop(bbox)
        mask = mask.crop(bbox)
        a = mask.split()[0].convert('L').point(
            lambda x: 255 if x > threshold else 0)
        mask = Image.merge('RGBA', (a, a, a, a.convert('L')))
        image_size = img.size
        info += image_size
        width = image_size[0]
        height = image_size[1]
        if(width != height):
            bigside = width if width > height else height
            background = Image.new('RGBA', (bigside, bigside), (0, 0, 0, 0))
            background2 = Image.new('RGBA', (bigside, bigside), (0, 0, 0, 0))
            offset = (int(round(((bigside - width) / 2), 0)),
                      int(round(((bigside - height) / 2), 0)))
            background.paste(img, offset)
            background2.paste(mask, offset)
            img = background
            mask = background2
        return img, mask, info

    def restore_by_file(self, raw, img, ref_img, blur_mask, info):
        raw_size = raw.size
        reshape_size = ref_img.size[0]
        upper_left_x = info[0]
        upper_left_y = info[1]
        img = img.resize(ref_img.size)
        bbox = ref_img.split(
        )[-1].convert('L').point(lambda x: 255 if x > 0 else 0, mode='1').getbbox()
        bbox = list(bbox)
        img = img.crop(bbox).convert('RGBA')
        resize_offset_x, resize_offset_y = -20, -20
        pos_offset_x, pos_offset_y = 10, 10
        if info[0] == 0:
            resize_offset_x += 10
            pos_offset_x -= 10
        if info[1] == 0:
            resize_offset_y += 10
            pos_offset_y -= 10
        if info[2] == raw_size[0]:
            resize_offset_x += 10
        if info[3] == raw_size[1]:
            resize_offset_y += 10

        blur_mask = blur_mask.resize(ref_img.size)
        blur_mask = blur_mask.crop(bbox).convert('RGBA')
        blur_img = Image.new('RGBA', (raw_size[0], raw_size[1]), (0, 0, 0, 0))
        blur_img.paste(blur_mask, (upper_left_x, upper_left_y))
        blur_img = blur_img.filter(ImageFilter.GaussianBlur(5))

        new_img = Image.new('RGBA', (raw_size[0], raw_size[1]), (0, 0, 0, 0))
        new_img.paste(img, (upper_left_x, upper_left_y))
        new_img = Image.alpha_composite(raw.convert('RGBA'), new_img)
        new_img.putalpha(blur_img.split()[-1].convert('L'))
        new_img = Image.alpha_composite(raw.convert('RGBA'), new_img)
        return new_img
