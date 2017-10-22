import argparse
import style_transfer_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument('content', type=str,
                        help='Content image file path')
    parser.add_argument('style', type=str,
                        help='Style image file path')
    parser.add_argument('out_dir', type=str,
                        help='Output directory path')
    parser.add_argument('--model', '-m', default='vgg19.model',
                        help='model file path')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--iter', default=2000, type=int,
                        help='number of iteration')
    parser.add_argument('--save-iter', default=100, type=int,
                        help='number of iteration for saving images')
    parser.add_argument('--lr', default=1.0, type=float,
                        help='learning rate')
    parser.add_argument('--content-weight', default=5, type=float,
                        help='content image weight')
    parser.add_argument('--style-weight', default=100, type=float,
                        help='style image weight')
    parser.add_argument('--tv-weight', default=1e-3, type=float,
                        help='total variation weight')
    parser.add_argument('--width', '-w', default=256, type=int,
                        help='image width, height')
    parser.add_argument('--method', default='gram', type=str, choices=['gram', 'mrf'],
                        help='style transfer method')
    parser.add_argument('--content-layers', default=['relu4_2'], type=str, nargs='+',
                        help='content layer names')
    parser.add_argument('--style-layers', default=['pool1', 'conv3_2'], type=str, nargs='+',
                        help='style layer names')
    parser.add_argument('--initial-image', default='random', type=str, choices=['content', 'random'],
                        help='initial image')
    parser.add_argument('--octave', default=5, type=int,
                        help='the number of octaves')
    parser.add_argument('--content-scale', default=1, type=float,
                        help='scale of content image')
    parser.add_argument('--keep-color', action='store_true',
                        help='keep image color')
    parser.add_argument('--match-color-histogram', action='store_true',
                        help='use matching color histogram algorithm')
    parser.add_argument('--luminance-only', action='store_true',
                        help='use luminance only algorithm')
    args = parser.parse_args()

    style_transfer_runner.run(args)
