import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
import x2ct_dataset
from utils import save_model
from utils import visualize
import params

from torch.utils.tensorboard import SummaryWriter   

# Source : 0, Target :1
source_test_loader = x2ct_dataset.x_train_loader_val
target_test_loader = x2ct_dataset.ct_train_loader_val

writer = SummaryWriter('./log')

# def source_only(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name):
#     print("Source-only training")
#     for epoch in range(params.epochs):
#         print('Epoch : {}'.format(epoch))

#         encoder = encoder.train()
#         classifier = classifier.train()
#         discriminator = discriminator.train()

#         classifier_criterion = nn.CrossEntropyLoss().cuda()

#         start_steps = epoch * len(source_train_loader)
#         total_steps = params.epochs * len(target_train_loader)
        
#         optimizer = optim.SGD(
#             list(encoder.parameters()) +
#             list(classifier.parameters()),
#             lr=0.01, momentum=0.9)
        
#         for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
#             source_image, source_label = source_data
#             p = float(batch_idx + start_steps) / total_steps

#             source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
#             source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

#             optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
#             optimizer.zero_grad()

#             source_feature = encoder(source_image)

#             # Classification loss
#             class_pred = classifier(source_feature)
#             class_loss = classifier_criterion(class_pred, source_label)

#             class_loss.backward()
#             optimizer.step()
#             if (batch_idx + 1) % 50 == 0:
#                 print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image), len(source_train_loader.dataset), 100. * batch_idx / len(source_train_loader), class_loss.item()))

#         if (epoch + 1) % 10 == 0:
#             test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='source_only')
#     save_model(encoder, classifier, discriminator, 'source', save_name)
#     visualize(encoder, 'source', save_name)

def tensorboard(encoder, classifier, target_test_loader, target_train_loader, epoch):
    encoder.cuda()
    classifier.cuda()
    
    target_correct = 0
    for batch_idx, target_data in enumerate(target_train_loader):
        # Target input -> Target Classification
        target_image, target_label = target_data
        target_image, target_label = target_image.cuda(), target_label.cuda()
        target_feature = encoder(target_image)
        target_output = classifier(target_feature)
        target_pred = target_output.data.max(1, keepdim=True)[1]
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).cpu().sum()
    acc_train = 100. * target_correct.item() / len(target_train_loader.dataset)
    
    target_correct = 0
    for batch_idx, target_data in enumerate(target_test_loader):
        # Target input -> Target Classification
        target_image, target_label = target_data
        target_image, target_label = target_image.cuda(), target_label.cuda()
        target_feature = encoder(target_image)
        target_output = classifier(target_feature)
        target_pred = target_output.data.max(1, keepdim=True)[1]
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).cpu().sum()
    acc_test = 100. * target_correct.item() / len(target_test_loader.dataset)
    
    writer.add_scalars('Accuracy', {
            'Train': acc_train,
            'Val': acc_test
        }, epoch)
    writer.flush()
    


def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name):
    print("DANN training")
    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))

        encoder = encoder.train()
        classifier = classifier.train()
        discriminator = discriminator.train()

        classifier_criterion = nn.CrossEntropyLoss().cuda()
        discriminator_criterion = nn.CrossEntropyLoss().cuda()

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)
        
        optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(classifier.parameters()) +
            list(discriminator.parameters()),
            lr=0.01,
            momentum=0.9)
        
        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data  

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

#             source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            
            
            combined_image = torch.cat((source_image, target_image), 0)
            # Auge label
            combined_label = torch.cat((source_label, target_label), 0)

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)


            # 1.Classification loss
            # class_pred = classifier(source_feature)
            # class_loss = classifier_criterion(class_pred, source_label)

            # Auge
            class_pred = classifier(combined_feature)
            
#             print('PRED: ', class_pred.size())
#             print('LABEL: ', combined_label.size())
            class_loss = classifier_criterion(class_pred, combined_label)



            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            total_loss = class_loss + domain_loss
            total_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    batch_idx * len(target_image), len(target_train_loader.dataset), 100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(), domain_loss.item()))

        if (epoch + 1) % 10 == 0:
            test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='dann')
        
        tensorboard(encoder, classifier, target_test_loader, target_train_loader, epoch)

    save_model(encoder, classifier, discriminator, 'source', save_name)
    visualize(encoder, 'source', save_name)
