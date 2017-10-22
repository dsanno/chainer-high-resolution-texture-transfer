import argparse
import numpy as np
import os
from PIL import Image
import six
import chainer
from chainer import functions as F
from chainer import cuda, optimizers, serializers
import util

from neural_style import NeuralStyle
from net import VGG19
from lbfgs import LBFGS

def open_and_resize_images(path, target_widths, model):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    result = []
    for target_width in target_widths:
        target_height = int(round(float(height * target_width) / width))
        target_image = image.resize((target_width, target_height), Image.BILINEAR)
        result.append(np.expand_dims(model.preprocess(np.asarray(target_image, dtype=np.float32), input_type='RGB'), 0))
    return result

def run(args):
    if args.out_dir != None:
        if not os.path.exists(args.out_dir):
            try:
                os.mkdir(args.out_dir)
            except:
                print 'cannot make directory {}'.format(args.out_dir)
                exit()
        elif not os.path.isdir(args.out_dir):
            print 'file path {} exists but is not directory'.format(args.out_dir)
            exit()
    vgg = VGG19()
    content_image = open_and_resize_images(args.content, [args.width], vgg)[0]
    content_height, content_width = content_image.shape[-2:]
    content_height = int(content_height * args.content_scale)
    content_width = int(content_width * args.content_scale)
    content_size = (content_height, content_width)
    print 'loading content image completed'
    width_list = [args.width / 2 ** i for i in six.moves.range(args.octave)]
    style_images = open_and_resize_images(args.style, width_list, vgg)
    if args.match_color_histogram:
        style_images = [util.match_color_histogram(style_image, content_image) for style_image in style_images]
    if args.luminance_only:
        content_image, content_iq = util.split_bgr_to_yiq(content_image)
        style_image, style_iq = util.split_bgr_to_yiq(style_images[0])
        content_mean = np.mean(content_image, axis=(1,2,3), keepdims=True)
        content_std = np.std(content_image, axis=(1,2,3), keepdims=True)
        style_mean = np.mean(style_image, axis=(1,2,3), keepdims=True)
        style_std = np.std(style_image, axis=(1,2,3), keepdims=True)
        style_images = [(style_image - style_mean) / style_std * content_std + content_mean for style_image in style_images]
    print 'loading style image completed'
    serializers.load_npz(args.model, vgg)
    print 'loading neural network model completed'
    optimizer = LBFGS(args.lr, stack_size=10)
    content_layers = args.content_layers
    style_layers = args.style_layers

    def on_epoch_done(epoch, x, losses):
        if (epoch + 1) % args.save_iter == 0:
            image = cuda.to_cpu(x.data)
            if args.luminance_only:
                image = util.join_yiq_to_bgr(image, content_iq)
            image = vgg.postprocess(image[0], output_type='RGB').clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(os.path.join(args.out_dir, 'out_{0:04d}.png'.format(epoch + 1)))
            print 'epoch {} done'.format(epoch + 1)
            print 'losses:'
            label_width = max(map(lambda (name, loss): len(name), losses))
            for name, loss in losses:
                print '  {0:{width}s}: {1:f}'.format(name, loss, width=label_width)

    model = NeuralStyle(vgg, optimizer, args.content_weight, args.style_weight, args.tv_weight,
            content_layers, style_layers, args.gpu, initial_image=args.initial_image,
            keep_color=args.keep_color)
    out_image = model.fit(content_image, style_images, args.iter, content_size, on_epoch_done)
    out_image = cuda.to_cpu(out_image.data)
    if args.luminance_only:
        out_image = util.join_yiq_to_bgr(out_image, content_iq)
    image = vgg.postprocess(out_image[0], output_type='RGB').clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(os.path.join(args.out_dir, 'out.png'))
