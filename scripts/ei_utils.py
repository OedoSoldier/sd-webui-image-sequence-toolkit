import re
import pandas as pd
from modules.processing import StableDiffusionProcessingImg2Img

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def gr_show_value_none(visible=True):
    return {"value": None, "visible": visible, "__type__": "update"}


def gr_show_and_load(value=None, visible=True):
    if value:
        if value.orig_name.endswith('.csv'):
            value = pd.read_csv(value.name)
        else:
            value = pd.read_excel(value.name)
    else:
        visible = False
    return {"value": value, "visible": visible, "__type__": "update"}


def gr_set_value(value=None, visible=True):
    return {"value": value, "visible": visible, "__type__": "update"}


def sort_images(lst):
    pattern = re.compile(r"\d+(?=\.)(?!.*\d)")
    return sorted(lst, key=lambda x: int(re.search(pattern, x).group()))


def I2I_Generator_Create(p, i2i_sample, i2i_mask_blur, full_res_inpainting, inpainting_padding, init_image, denoise, cfg, steps, width, height, tiling, scripts, scripts_list, alwaysonscripts_list, script_args, positive, negative):
    i2i = StableDiffusionProcessingImg2Img(
                init_images = [init_image],
                resize_mode = 0,
                denoising_strength = 0,
                mask = None,
                mask_blur= i2i_mask_blur,
                inpainting_fill = 1,
                inpaint_full_res = full_res_inpainting,
                inpaint_full_res_padding= inpainting_padding,
                inpainting_mask_invert= 0,
                sd_model=p.sd_model,
                outpath_samples=p.outpath_samples,
                outpath_grids=p.outpath_grids,
                restore_faces=p.restore_faces,
                prompt='',
                negative_prompt='',
                styles=p.styles,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=i2i_sample,
                n_iter=1,
                batch_size=1,
                steps=steps,
                cfg_scale=cfg,
                width=width,
                height=height,
                tiling=tiling,
            )
    i2i.denoising_strength = denoise
    i2i.do_not_save_grid = True
    i2i.do_not_save_samples = True
    i2i.override_settings = {}
    i2i.override_settings_restore_afterwards = {}
    i2i.scripts = scripts
    i2i.scripts.scripts = scripts_list.copy()
    i2i.scripts.alwayson_scripts = alwaysonscripts_list.copy()
    i2i.script_args = script_args
    i2i.prompt = positive
    i2i.negative_prompt = negative
    
    return i2i