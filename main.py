from __future__ import print_function

import os
import matplotlib.pyplot as plot
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from option import Options
from utils import get_torch_vars, compute_loss # run_validation_step
from loss import CrossEntropyLabelSmooth
from tqdm import tqdm
from PIL import Image
import time
import numpy as np
import cv2
from models import simple, Zhang_model
from dataset import *
import pickle

# global variable
best_pred = 100.0
errlist_train = []
errlist_val = []


def main():
    # init the args
    args = Options().parse()
    args.no_cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    args.plot = False
    args.dataset = 'data'
    args.model = 'UNet'
    args.batch_size = 128  # 128 #32
    args.loss = 'CrossEntropyLoss'
    args.backbone = 'resnet18'

    args.lr = 0.001  # 0.004  #0.01 #
    args.epochs = 5  # 60#20 #60
    args.lr_step = 15
    args.test_aug = False
    args.nclass = 24
    args.ohem = -2
    args.atte = False
    args.basepath = './'
    args.classification = False
    args.skip_connection = False

    # plot
    if args.plot:
        print('=>Enabling matplotlib for display:')
        plot.ion()
        plot.show()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # IMAGE DATA
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print(x_train.shape)

    # Transform data into rgb:
    # print("Transforming RGB data into Lab color space")
    # train_L, train_ab = cvt2lab(x_train, classification=False, num_class=10)
    print("Load Training data from presaved .npy files")
    if args.classification:
        train_L = np.load('./data/train_L.npy')
        train_ab = np.load('./data/classification_train_ab.npy')
        train_a = train_ab[0]
        train_b = train_ab[1]
    else:
        train_L = np.load('./data/train_L.npy')
        train_ab = np.load('./data/regression_train_ab.npy')
    print(train_ab.shape)
    print(train_L.shape)
    # print(train_ab.shape)

    # train_rgb_cat = get_rgb_cat(train_rgb, colours)
    # test_L, _ = cvt2lab(x_test, classification=False, num_class=10)
    # np.save('./data/test_L.npy', test_L)
    test_L = np.load('./data/test_L.npy')
    # test_rgb_cat = get_rgb_cat(test_rgb, colours)

    # train_loader, test_loader = Dataloder(args).getloader()

    # init the model
    if args.model == 'UNet':
        model = simple(args)
    # print(model)

    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'CrossEntropyLabelSmooth':
        criterion = CrossEntropyLabelSmooth(num_classes=args.nclass, ignore_index=-1)
    criterion = nn.MSELoss()



    # optimizer = get_optimizer(args, model, False)
    # if args.cuda:
    #     model.cuda()
    #     model = torch.nn.DataParallel(model)
    # # check point
    #
    # if args.resume is not None:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch'] + 1
    #         best_pred = checkpoint['best_pred']
    #         # errlist_train = checkpoint['errlist_train']
    #         # errlist_val = checkpoint['errlist_val']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #         for name, param in checkpoint['state_dict'].items():
    #             print(name)
    #     else:
    #         print("=> no resume checkpoint found at '{}'". \
    #               format(args.resume))
    #
    # if args.eval == False:
    #     scheduler = LR_Scheduler(args, len(train_loader))

    def train():
        model.train()
        model.cuda()
        print("Beginning training ...")
        start = time.time()
        train_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            losses = []
            for i in range(0, train_L.shape[0], args.batch_size):
                batch_ab = torch.autograd.Variable(torch.from_numpy(train_ab[i:i+args.batch_size]).float().cuda(), requires_grad = False)
                batch_grey = torch.autograd.Variable(torch.from_numpy(train_L[i:i+args.batch_size]).float().cuda(), requires_grad = False)
                optimizer.zero_grad()
                batch_output = model(batch_grey)
                loss = criterion(batch_output, batch_ab)
                loss.backward()
                optimizer.step()
                losses.append(loss.data.item())
            avg_loss = np.mean(losses)
            train_loss.append(avg_loss)
            # save_dir = "outputs/" + args.experiment_name
            # # Create the outputs folder if not created already
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, args.epochs, avg_loss))
        return model
    def test():
        model = simple(args)
        model.load_state_dict(torch.load('./checkpoints/final_model.pth'))
        model.eval()
        test_L_input = torch.autograd.Variable(torch.from_numpy(test_L[0:20]).float(), requires_grad=False)
        test_ab = model(test_L_input)
        RGB = cvt2RGB(test_L_input.data.numpy(), test_ab.data.numpy())
        RGB *= 255
        print(RGB.shape)
        np.save('./generated_RGB_data.npy', RGB)
    test()
    # final_model = train()
    # torch.save(final_model.state_dict(), './checkpoints/final_model.pth')

    #     print("Beginning training ...")
    #     start = time.time()
    #     train_losses = []
    #     valid_losses = []
    #     valid_accs = []
    #     for epoch in range(args.epochs):
    #         losses = []
    #         for i, (xs, ys) in enumerate(get_batch(train_grey,
    #                                                train_rgb_cat,
    #                                                args.batch_size)):
    #             images, labels = get_torch_vars(xs, ys, args.gpu)
    #             # Forward + Backward + Optimize
    #             optimizer.zero_grad()
    #             outputs = model(images)
    #
    #             loss = compute_loss(criterion,
    #                                 outputs,
    #                                 labels,
    #                                 batch_size=args.batch_size,
    #                                 num_colours=num_colours)
    #             loss.backward()
    #             optimizer.step()
    #             losses.append(loss.data.item())
    #     if args.plot:
    #         _, predicted = torch.max(outputs.data, 1, keepdim=True)
    #         plot(xs, ys, predicted.cpu().numpy(), colours,
    #              save_dir+'/train_%d.png' % epoch,
    #              args.visualize,
    #              args.downsize_input)
    #     avg_loss = np.mean(losses)
    #     train_losses.append(avg_loss)
    #     time_elapsed = time.time() - start
    #     print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (epoch+1, args.epochs, avg_loss, time_elapsed))
    #     model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    #     val_loss, val_acc = run_validation_step(model,
    #                                             criterion,
    #                                             test_grey,
    #                                             test_rgb_cat,
    #                                             args.batch_size,
    #                                             colours,
    #                                             save_dir+'/test_%d.png' % epoch,
    #                                             args.visualize,
    #                                             args.downsize_input)
    #
    #     time_elapsed = time.time() - start
    #     valid_losses.append(val_loss)
    #     valid_accs.append(val_acc)
    #     print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %.2f' % (epoch+1, args.epochs, val_loss, val_acc, time_elapsed))
    #
    #     plot.figure()
    #     plot.plot(train_losses, "ro-", label="Train")
    #     plot.plot(valid_losses, "go-", label="Validation")
    #     plot.legend()
    #     plot.title("Loss")
    #     plot.xlabel("Epochs")
    #     plot.savefig(save_dir + "/training_curve.png")
    #
    #     if args.checkpoint:
    #         print('Saving model...')
    #         torch.save(model.state_dict(), args.checkpoint)
    #
    #     return model
    # train()
    #
    #     train_losses = []
    #     valid_losses = []
    #     valid_accs = []
    #     for epoch in range(args.epochs):
    #         # Train the Model
    #         losses = []
    #         for i, (xs, ys) in enumerate(get_batch(train_grey,
    #                                                train_rgb_cat,
    #                                                args.batch_size)):
    #             images, labels = get_torch_vars(xs, ys, args.gpu)
    #             # Forward + Backward + Optimize
    #             optimizer.zero_grad()
    #             outputs = model(images)
    #
    #             loss = compute_loss(criterion,
    #                                 outputs,
    #                                 labels,
    #                                 batch_size=args.batch_size,
    #                                 num_colours=num_colours)
    #             loss.backward()
    #             optimizer.step()
    #             losses.append(loss.data.item())
    #
    #     # plot training images
    #     if args.plot:
    #         _, predicted = torch.max(outputs.data, 1, keepdim=True)
    #         plot(xs, ys, predicted.cpu().numpy(), colours,
    #              save_dir+'/train_%d.png' % epoch,
    #              args.visualize,
    #              args.downsize_input)
    #
    #     # plot training images
    #     avg_loss = np.mean(losses)
    #     train_losses.append(avg_loss)
    #     time_elapsed = time.time() - start
    #     print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (
    #         epoch+1, args.epochs, avg_loss, time_elapsed))
    #
    #     # Evaluate the model
    #     model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    #     val_loss, val_acc = run_validation_step(cnn,
    #                                             criterion,
    #                                             test_grey,
    #                                             test_rgb_cat,
    #                                             args.batch_size,
    #                                             colours,
    #                                             save_dir+'/test_%d.png' % epoch,
    #                                             args.visualize,
    #                                             args.downsize_input)
    #
    #     time_elapsed = time.time() - start
    #     valid_losses.append(val_loss)
    #     valid_accs.append(val_acc)
    #     print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %.2f' % (
    #         epoch+1, args.epochs, val_loss, val_acc, time_elapsed))
    #
    #     plot.figure()
    #     plot.plot(train_losses, "ro-", label="Train")
    #     plot.plot(valid_losses, "go-", label="Validation")
    #     plot.legend()
    #     plot.title("Loss")
    #     plot.xlabel("Epochs")
    #     plot.savefig(save_dir + "/training_curve.png")
    #
    #     if args.checkpoint:
    #         print('Saving model...')
    #         torch.save(model.state_dict(), args.checkpoint)
    #
    #     return model

        # tbar = tqdm(train_loader, desc='\r')
        # for batch_idx, (data, target, ids) in enumerate(tbar):
        #     scheduler(optimizer, batch_idx, epoch, best_pred)
        #
        #     if args.cuda:
        #         data, target = data.cuda(), target.cuda()
        #
        #     data, target = Variable(data), Variable(target)
        #     optimizer.zero_grad()
        #     output, f, center_weight, softmax_out = model(data)
        #
        #     # May need Ohem later.
        #     # Here leave it for future usage.
        #     if args.ohem > -1:
        #         if args.loss == 'CrossEntropyLoss':
        #             loss = nn.CrossEntropyLoss(reduce=False)(output, target).cpu().detach().numpy()
        #         elif args.loss == 'CrossEntropyLabelSmooth':
        #             loss = CrossEntropyLabelSmooth(num_classes=args.nclass, reduce=False)(output, target).cpu().detach().numpy()
        #         else:
        #             raise Keyerror('Not implement!')
        #         loss_index = np.argsort(loss)
        #         targets_copy = target.clone()
        #         if args.nclass < 3:
        #             targets_copy[loss_index[0:args.ohem]] = -1
        #
        #     else:
        #         targets_copy = target.clone()
        #
        #     loss = criterion(output, targets_copy)
        #
        #     loss.backward()
        #     optimizer.step()
        #
        #     train_loss += loss.data
        #     pred = output.data.max(1)[1]
        #     correct += pred.eq(target.data).cpu().sum()
        #     total += target.size(0)
        #     err = 100. - 100. * correct / total
        #     tbar.set_description('\rLoss: %.3f | Err: %.3f%% (%d/%d)' % \
        #                          (train_loss / (batch_idx + 1), err, total - correct, total))
        #     train_loss_end = train_loss
        #     batch_idx_end = batch_idx
        #
        # print('\rLoss: %.3f | Err: %.3f%% (%d/%d)' % \
        #       (train_loss / (batch_idx_end + 1), err, total - correct, total))

    # Place holder
    # Test function has not been implemented. Wait for later finishing
    # Here only leave a skeleton code.
    # def test(epoch):
    #     model.eval()
    #     global best_pred, errlist_train, errlist_val
    #     test_loss, correct, total = 0, 0, 0
    #     is_best = False
    #     tbar = tqdm(test_loader, desc='\r')
    #
    #     batch_idx_end = 0
    #
    #     TP = 0
    #     FN = 0
    #     FP = 0
    #     TN = 0
    #
    #     with torch.no_grad():
    #         for batch_idx, (data, target, ids) in enumerate(tbar):
    #             if args.cuda:
    #                 data, target = data.cuda(), target.cuda()
    #             data, target = Variable(data, volatile=True), Variable(target)
    #             placeholder = model(data)
    #             test_loss += 0  # criterion(output, target).data

                # pred = output.data.max(1)[1]
                #
                # is_right = pred.eq(target.data).cpu().sum()
                # correct += is_right
                # total += target.size(0)
                #
                # err = 100. - 100. * correct / total
                # tbar.set_description('Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                #                      (test_loss / (batch_idx + 1), err, total - correct, total))
                # batch_idx_end = batch_idx

        # recall_1 = TP / (TP + FN)
        # precision_1 = TP / (TP + FP)
        # recall_0 = TN / (TN + FP)
        # precision_0 = TN / (TN + FN)

    #     print(
    #         'Loss: %.3f | Err: %.3f%%, Recall_1:%.3f%%, Precision_1:%.3f%%, Recall_0:%.3f%%, Precision_0:%.3f%%, (%d/%d)' % \
    #         (
    #         test_loss / (batch_idx_end + 1), err, recall_1, precision_1, recall_0, precision_0, total - correct, total))
    #     print('TN: %d | TP: %d | FN: %d | FP: %d' % (TN, TP, FN, FP))
    #     if args.eval:
    #         print('Error rate is %.3f%%' % err)
    #         return
    #     # save checkpoint
    #     errlist_val += [err]
    #     if err < best_pred:
    #         best_pred = err
    #         is_best = True
    #     print(err)
    #     print('Best Error rate is %.3f%%' % best_pred)
    #
    #     time.sleep(10)
    #     save_checkpoint({
    #         'epoch': epoch,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'best_pred': best_pred,
    #         'errlist_train': errlist_train,
    #         'errlist_val': errlist_val,
    #     }, args=args, is_best=is_best)
    #
    #     if args.plot:
    #         plot.clf()
    #         plot.xlabel('Epoches: ')
    #         plot.ylabel('Error Rate: %')
    #         plot.plot(errlist_train, label='train')
    #         plot.plot(errlist_val, label='val')
    #         plot.legend(loc='upper left')
    #         plot.draw()
    #         plot.pause(0.001)
    #
    # if args.eval:
    #     test(args.start_epoch)
    #     return
    #
    # for epoch in range(args.start_epoch, args.epochs + 1):
    #     train(epoch)
    #     test(epoch)
    #
    # # save train_val curve to a file
    # if args.plot:
    #     plot.clf()
    #     plot.xlabel('Epoches: ')
    #     plot.ylabel('Error Rate: %')
    #     plot.plot(errlist_train, label='train')
    #     plot.plot(errlist_val, label='val')
    #     plot.savefig("%s/runs/%s/%s/%s/" % (args.basepath, args.dataset, args.model, args.checkname)
    #                  + 'train_val.jpg')
if __name__ == "__main__":
    main()
