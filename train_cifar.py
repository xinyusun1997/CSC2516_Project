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
from utils import get_torch_vars, compute_loss, evaluation_metrics, ssim_compute, psnr_compute # run_validation_step
from loss import CrossEntropyLabelSmooth
import time
import numpy as np
import cv2
from models import simple, NNEncLayer, decode, simple_skip_connection
from dataset import *
import pickle

def main():
    # init the args
    args = Options().parse()
    args.no_cuda = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    args.plot = False
    args.dataset = 'data'
    args.batch_size = 128  # 128 #32
    args.loss = 'CrossEntropyLoss'
    args.backbone = 'resnet18'

    args.lr = 1e-4  # 0.004  #0.01 #
    args.epochs = 20  # 60#20 #60
    args.lr_step = 15

    args.basepath = './'

    args.classification = True
    args.skip_connection = True
    args.eval = False
    args.from_npy = True


    print('Load data from saved .npy files...')
    print('Note for landscape data, these are Lab type.')
    # (x_train, y_train), (x_test, y_test) = load_cifar10()
    # train_L, train_ab = cvt2lab_new(x_train, classification=args.classification, num_class=10)
    # np.save('./data/classification_train_ab_new.npy', train_ab)
    # return
    # Make .npy files
    # This will take long time.
    if not args.from_npy:
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        print("Transforming RGB data into Lab color space")
        train_L, train_ab = cvt2lab_new(x_train, classification=args.classification, num_class=10)
        np.save('./data/train_L.npy', train_L)
        if args.classification:
            np.save('./data/classification_train_ab_new.npy', train_ab)
        else:
            np.save('./data/regession_train_ab.npy', train_ab)

        test_rgb = np.rollaxis(x_test, 1, 4)
        test_rgb = test_rgb / 255
        np.save('./data/test_rgb.npy', test_rgb)
        test_L, _ = cvt2lab(x_test, classification=False, num_class=10)
        np.save('./data/test_L.npy', test_L)

    # IMAGE DATA
    # Read from saved .npy files
    else:
        if not args.eval:
            print("Load Training data from saved .npy files")
            if args.classification:
                train_L = np.load('./data/train_L.npy')
                train_ab = np.load('./data/classification_train_ab_new.npy')
                train_a = train_ab[0]
                train_b = train_ab[1]
            else:
                train_L = np.load('./data/train_L.npy')
                train_ab = np.load('./data/regression_train_ab.npy')
        test_L = np.load('./data/test_L.npy')
        test_RGB_gt = np.load('./data/test_rgb.npy')
    print('Finish Data Loading')

    # init the model
    model = simple(args)
    if args.cuda:
        model.cuda()

    if args.classification:
        criterion = nn.CrossEntropyLoss()
        model_path = './checkpoints/classification_best_model.pth'
    else:
        criterion = nn.MSELoss()
        model_path = './checkpoints/regression_best_model.pth'

    def train(model):
        print("Beginning training ...")
        train_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_loss = float('inf')
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for i in range(0, train_L.shape[0], args.batch_size):
                # Regression Training
                if not args.classification:
                    batch_grey = torch.autograd.Variable(torch.from_numpy(train_L[i:i+args.batch_size]).float(), requires_grad = False)
                    batch_ab = torch.autograd.Variable(torch.from_numpy(train_ab[i:i + args.batch_size]).float(),requires_grad=False)
                    if args.cuda:
                        batch_ab = batch_ab.cuda()
                        batch_grey = batch_grey.cuda()
                    optimizer.zero_grad()
                    batch_output = model(batch_grey)
                    loss = criterion(batch_output, batch_ab)
                # Classification Training
                else:
                    batch_grey = torch.autograd.Variable(torch.from_numpy(train_L[i:i+args.batch_size]).float(), requires_grad = False)
                    batch_a = torch.autograd.Variable(torch.from_numpy(train_a[i:i + args.batch_size]).long(), requires_grad=False)
                    batch_b = torch.autograd.Variable(torch.from_numpy(train_b[i:i + args.batch_size]).long(), requires_grad=False)
                    if args.cuda:
                        batch_grey = batch_grey.cuda()
                        batch_a = batch_a.cuda()
                        batch_b = batch_b.cuda()
                    optimizer.zero_grad()
                    batch_output = model(batch_grey)
                    loss_a = criterion(batch_output[0], batch_a)
                    loss_b = criterion(batch_output[1], batch_b)
                    loss = loss_a + loss_b
                loss.backward()
                optimizer.step()
                losses.append(loss.data.item())
            avg_loss = np.mean(losses)
            train_loss.append(avg_loss)
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, args.epochs, avg_loss))

            if avg_loss < best_loss:

                torch.save(model.state_dict(), model_path)
                best_loss = avg_loss
                print('Best Training Model Saved')
        return model

    def test(model):
        model.eval()
        eval_loss_test = []
        for i in range(0, test_L.shape[0], args.batch_size):
            test_L_input = torch.autograd.Variable(torch.from_numpy(test_L[i:i+args.batch_size]).float(), requires_grad=False)
            if args.cuda:
                test_L_input = test_L_input.cuda()
            test_ab = model(test_L_input)
            if not args.classification:
                if args.cuda:
                    pred_rgb = cvt2RGB(test_L_input.cpu().data.numpy(), test_ab.cpu().data.numpy(), args.classification)
                else:
                    pred_rgb = cvt2RGB(test_L_input.data.numpy(), test_ab.data.numpy(), args.classification)
            else:
                if args.cuda:
                    test_ab = (test_ab[0].cpu().data.numpy(), test_ab[1].cpu().data.numpy())
                    pred_rgb = cvt2RGB(test_L_input.cpu().data.numpy(), test_ab, args.classification)
                else:
                    test_ab = (test_ab[0].data.numpy(), test_ab[1].data.numpy())
                    pred_rgb = cvt2RGB(test_L_input.data.numpy(), test_ab, args.classification)
            # np.save('./pred_rgb_example.npy', pred_rgb)
            gt_rgb = test_RGB_gt[i:i+args.batch_size]
            eval_loss = evaluation_metrics(pred_rgb, gt_rgb)
            eval_loss_test.append(eval_loss)
        eval_loss = np.mean(eval_loss_test)
        print('Average Deviation Level in Color Saturation is %.4f ' % eval_loss)

    if not args.eval:
        model = train(model)
    else:
        model.load_state_dict(torch.load(model_path))
        test(model)


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

    # torch.save(final_model.state_dict(), './checkpoints/final_model.pth')
    # test()
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
