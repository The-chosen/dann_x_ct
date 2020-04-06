import torch
import train
import x2ct_dataset
import x2ct_model
from utils import get_free_gpu

save_name = 'omg'


def main():
    print('Start load data...')
    source_train_loader = x2ct_dataset.x_train_loader_train
    target_train_loader = x2ct_dataset.ct_train_loader_train

    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = x2ct_model.Extractor().cuda()
        classifier = x2ct_model.Classifier().cuda()
        discriminator = x2ct_model.Discriminator().cuda()

        train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name)

    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    print('hello')
    main()
