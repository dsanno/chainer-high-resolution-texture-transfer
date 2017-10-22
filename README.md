# High Resolution Texture Transfer
Implementation of Neural Style Transfer using [High-Resolution Multi-Scale Neural Texture Synthesis](https://wxs.ca/research/multiscale-neural-synthesis/)

# Requirements

* Python 2.7
* [Chainer 2.0.0](http://chainer.org/)
* [Cupy 1.0.0](http://docs.cupy.chainer.org/en/stable/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)

# Usage

## Download VGG 19 layers caffe model

* Visit https://gist.github.com/ksimonyan/3785162f95cd2d5fee77 and download VGG_ILSVRC_16_layers.caffemodel.
* Put downloaded file into this directory.

## Convert caffemodel to chainer model

```
$ python src/create_chainer_model.py
```

## Transfer image style using "A Neural Algorithm of Artistic Style"

```
$ python src/run.py content_image style_image output_dir [options]
```

Example:
```
$ python src/run.py content_image.png style_image.png output/texture -g 0
```

Parameters:

* `content_image`: (Required) Content image file path
* `style_image`: (Required) Style image file path
* `output_dir`: (Required) Output directory path
* `-g (--gpu) <int>`: (Optional) GPU device index. Negative value indecates CPU (default: -1)
* `-w (--width) <int>`: (Optional) Image width (default: 256)
* `--iter <int>`: (Optional) Number of iteration for each iteration (default: 2000)
* `--initial-image <str>`: (Optional) Initial image of optimization: "random" or "content" (default: random)
* `--save-iter <int>`: (Optional) Learning rate (default: 1)
* `--content-layers <str> <str> ...`: (Optional) Layer names to use for content reconstruction (default: relu3_3 relu4_3)
* `--style-layers <str> <str> ...`: (Optional) Layer names to use for style reconstruction. (default: pool1 conv3_2)
* `--content-weight <float>`: (Optional) Weight of content loss (default: 5)
* `--style-weight <float>`: (Optional) Weight of style loss (default: 100)
* `--tv-weight <float>`: (Optional) Weight of total variation loss (default: 1e-3)

Experimantal options:

* `--keep-color`: (Optional) Keep color phase
* `--match-color-histogram`: (Optional) Use "Color histogram matching" algorithm in "Preserving Color in Neural Artistic Style Transfer"
* `--luminance-only`: (Optional) Use "Luminance-only" algorithm in "Preserving Color in Neural Artistic Style Transfer"

# Example

## Command

```
$ python src/run.py sample/tubingen.jpg sample/block.jpg output/synthesized -g 0 -w 800
```

## Content Image
![Content Image](/sample/stone.jpg)

## Style Image

![Style Image](/sample/cat.jpg)

## Synthesized Image

![Synthesized Image](/sample/synthesized.jpg)

# License

MIT License
