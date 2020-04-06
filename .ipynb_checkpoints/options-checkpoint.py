import argparse
import torchvision.models as models

class options():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__")

    and callable(models.__dict__[name]))
        self.opt = self.initialize(parser)

    def initialize(self, parser):
        parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)') 
        parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--extractor', metavar='ARCH', default='resnet18',
                    choices=self.model_names,
                    help='extractor architecture: ' +
                        ' | '.join(self.model_names) +
                        ' (default: resnet18)')
        parser.add_argument('--classifier', metavar='ARCH', default='resnet34',
                    choices=self.model_names,
                    help='classifier architecture: ' +
                        ' | '.join(self.model_names) +
                        ' (default: resnet18)')
        parser.add_argument('--discriminator', metavar='ARCH', default='resnet18',
                    choices=self.model_names,
                    help='discriminator architecture: ' +
                        ' | '.join(self.model_names) +
                        ' (default: resnet18)')
        return parser.parse_args()