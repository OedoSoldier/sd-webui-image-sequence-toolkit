# Enhanced img2img

This is an extension for [AUTOMATIC111's WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), support for batched and better inpainting.

## Install

See [official wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions).

## Usage

### Enhanced img2img

Switch to **"img2img"** tab, under the **"script"** column, select **"enhanced img2img"**.

![](screenshot_1.png)

  - **Input directory**: a folder containing all the images you want to process.
  - **Output directory**: a folder to save output images.
  - **Mask directory**: a folder containing all the masks. *Not essential*.
  - **Use input image's alpha channel as mask**: if your original images are PNG files with transparent backgrounds, you can use this option to create outputs with transparent backgrounds. *Note: when this option is selected, the masks in the "**mask directory**" will not be used.*
  - **Use another image as mask**: use masks in the "**mask directory**" to inpaint images. *Note: if the relevant masks are blank images or no mask is provided, the original images will not be processed.*
  - **Use mask as output alpha channel**: as it says. *Note: when the "**use input image's alpha channel as mask**" option is selected, this option is automatically activated.*
  - **Zoom in masked area**: crop and resize the masked area to square images; this will give better results when the masked area is relatively small compared to the original images.
  - **Alpha threshold**: alpha value for determine background and foreground.
  - **Rotate images (clockwise)**: as it says, this can improve AI's performance when the original images are upside down.
  - **Process given file(s) under the input folder, seperate by comma**: process certain image(s) from the text box right to it; if this option is not checked, all the images under the folder will be processed.
  - **Files to process**: filenames of images you want to process. I recommend naming your images with a digit suffixes, e.g. `000233.png, 000234.png, 000235.png, ...` or `image_233.jpg, image_234.jpg, image_235.jpg, ...`. In this way, you can use `233,234,235` or simply `233-235` to assign these files. Otherwise, you need to give the full filenames like `image_a.webp,image_b.webp,image_c.webp`.
  - **Use deepbooru prompt**: use DeepDanbooru to predict image tags; if you have input some prompts in the prompt area, it will append to the end of the prompts.
  - **Using contextual information**: only if tags are present in both current and next frames' prediction results, this can improve accuracy (maybe).
  - **Loopback**: similar to the loopback script, this will run input images img2img twice to enhance AI's creativity. 
  - **Firstpass width** and **firstpass height**: AI tends to be more creative when the firstpass size is smaller.
  - **Denoising strength**: denoising strength for the first pass, better be no higher than 0.4.
  - **Read tags from text files**: will read tags from text files with the same filename as the current input image.
  - **Text files directory**: Optional, will load from input dir if not specified
  - **Use csv prompt list** and **input file path**: use a `.csv` file as prompts for each image, one line for one image.

### Multi-frame rendering

Switch to **"img2img"** tab, under the **"script"** column, select **"multi-frame rendering"**, **should be used with ControlNet**. For more information, see: [the original post](https://xanthius.itch.io/multi-frame-rendering-for-stablediffusion).

![](screenshot_2.png)

  - **Input directory**: a folder containing all the images you want to process.
  - **Output directory**: a folder to save output images.
  - **Initial denoise strength**: the denoising strength of the first frame. You can set the noise reduction strength of the first frame and the rest of the frames separately. The noise reduction strength of the rest of the frames is controlled through the img2img main interface.
  - **Append interrogated prompt at each iteration**: use CLIP or DeepDanbooru to predict image tags; if you have input some prompts in the prompt area, it will append to the end of the prompts.
  - **Third frame (reference) image**: the image used to put at the third frame.
    - None: use only two images, the previous frame and the current frame, without a third reference image.
    - FirstGen: use the **processed** first image as the reference image.
    - OriginalImg: use the **original** first image as the reference image.
    - Historical: use the second-to-last frame before the current frame as the reference image.
  - **Enable color correction**: use color correction based on the loopback image. When using a non-FirstGen image as the reference image, turn on to reduce color fading.
  - **Unfreeze Seed**: once checked, the basic seed value will be incremented by 1 automatically each time an image is generated.
  - **Loopback Source**: the images in the second frame.
    - Previous: generates the image from the previous generated image.
    - Currrent: generates the image from the current image.
    - First: generates the image from the first generated image.
  - **Read tags from text files**: will read tags from text files with the same filename as the current input image.
  - **Text files directory**: Optional, will load from input dir if not specified
  - **Use csv prompt list** and **input file path**: use a `.csv` file as prompts for each image, one line for one image.

## Tutorial video (in Chinese)

<iframe src="//player.bilibili.com/player.html?aid=563344169&bvid=BV1pv4y1o7An&cid=911472358&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

<iframe src="//player.bilibili.com/player.html?aid=865839831&bvid=BV1R54y1M7u5&cid=1047760345&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

## Credit

AUTOMATIC1111's WebUI - https://github.com/AUTOMATIC1111/stable-diffusion-webui

Multi-frame Rendering - https://xanthius.itch.io/multi-frame-rendering-for-stablediffusion
