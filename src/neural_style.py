import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import Variable

import util

class NeuralStyle(object):
    def __init__(self, model, optimizer, content_weight, style_weight, tv_weight, content_layers, style_layers, device_id=-1, initial_image='random', keep_color=False):
        self.model = model
        self.optimizer = optimizer
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.device_id = device_id
        self.content_layer_names = content_layers
        self.style_layer_names = style_layers
        self.initial_image = initial_image
        self.keep_color = keep_color
        if device_id >= 0:
            self.xp = cuda.cupy
            self.model.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, image_size, content_image, style_image, epoch_num, callback=None):
        device_id = None
        if self.device_id >= 0:
            device_id = self.device_id
        with cuda.get_device_from_id(device_id):
            return self.__fit(image_size, content_image, style_image, epoch_num, callback)

    def __fit(self, content_image, style_images, epoch_num, content_size, callback=None):
        xp = self.xp
        input_image = None
        base_epoch = 0
        content_x = xp.asarray(content_image)
        content_layer_names = self.content_layer_names
        with chainer.using_config('enable_backprop', False):
            if content_x.shape[-2:] != content_size:
                h = F.resize_images(content_x, content_size)
            else:
                h = content_x
            content_layers = self.model(h, content_layer_names)
        style_layer_names = self.style_layer_names
        style_grams = []
        for style_image in style_images:
            if self.keep_color:
                style_x = util.luminance_only(xp.asarray(style_image), content_x)
            else:
                style_x = xp.asarray(style_image)
            with chainer.using_config('enable_backprop', False):
                style_layers = self.model(style_x, style_layer_names)
            style_grams.append({name: util.gram_matrix(layer) for name, layer in style_layers.items()})
        if input_image is None:
            if self.initial_image == 'content':
                input_image = xp.asarray(content_image[:,:])
            else:
                input_image = xp.random.normal(0, 1, size=content_x.shape).astype(np.float32) * 0.001
        else:
            input_image = input_image.repeat(2, 2).repeat(2, 3)
            h, w = content_x.shape[-2:]
            input_image = input_image[:,:,:h,:w]
        link = chainer.Link(x=input_image.shape)
        if self.device_id >= 0:
            link.to_gpu()
        link.x.data[:] = xp.asarray(input_image)
        self.optimizer.setup(link)
        for epoch in six.moves.range(epoch_num):
            loss_info = self.__fit_one(link, content_layers, style_grams, content_size)
            if callback:
                callback(base_epoch + epoch, link.x, loss_info)
        base_epoch += epoch_num
        input_image = link.x.data
        return link.x

    def __fit_one(self, link, content_layers, style_grams, content_size):
        xp = self.xp
        link.zerograds()
        height, width = link.x.shape[-2:]
        loss_info = []
        loss = Variable(xp.zeros((), dtype=np.float32))
        content_height, content_width = content_size
        same_content_size = content_height == height and content_width == width
        if not same_content_size or self.keep_color:
            if same_content_size:
                h = link.x
            else:
                h = F.resize_images(link.x, content_size)
            layer_names = content_layers.keys()
            layers = self.model(h, layer_names)
            for name, content_layer in content_layers.items():
                layer = layers[name]
                content_loss = self.content_weight * F.mean_squared_error(layer, content_layer)
                loss_info.append(('content_' + name, float(content_loss.data)))
                loss += content_loss
        for i, style_gram in enumerate(style_grams):
            h = F.resize_images(link.x, (height // 2 ** i, width // 2 ** i))
            if self.keep_color:
                layer_names = content_layers.keys()
                layers = self.model(util.gray(h), style_gram.keys())
            elif not same_content_size or i > 0:
                layer_names = style_gram.keys()
                layers = self.model(h, layer_names)
            else:
                layer_names = list(set(content_layers.keys() + style_gram.keys()))
                layers = self.model(h, layer_names)
                for name, content_layer in content_layers.items():
                    layer = layers[name]
                    content_loss = self.content_weight * F.mean_squared_error(layer, content_layer)
                    loss_info.append(('content_' + name, float(content_loss.data)))
                    loss += content_loss
            for name, style_gram in style_gram.items():
                gram = util.gram_matrix(layers[name])
                style_loss = self.style_weight * F.mean_squared_error(gram, style_gram)
                loss_info.append(('style{}_{}'.format(i + 1, name), float(style_loss.data)))
                loss += style_loss
        tv_loss = self.tv_weight * util.total_variation(link.x)
        loss_info.append(('tv', float(tv_loss.data)))
        loss += tv_loss
        loss.backward()
        self.optimizer.update()
        return loss_info
