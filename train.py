import argparse
import os
from math import log10

import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import autograd
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform

from model.model import Generator, Discriminator
from model.loss import Generator_loss


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# attention the crop_size must be 2^n because of the hourglass layer!!!!
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=1000, type=int, help='train epoch number')
parser.add_argument('--batchSize', default=64, type=int, help='train batchSize')
parser.add_argument('--testBatchSize', default=1, type=int, help='test batchSize')
parser.add_argument('--critic_iter', default=5, type=int, help='WGAN critic_iter')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--start', default=1, type=int, help='start epoch')
parser.add_argument('--pretrain_path', default="pretrain_netG_epoch_4_31.pth", type=str, help='pretrain_model_path')


def calculate_gradient_penalty(batch_size, netD, real_images, fake_images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = torch.FloatTensor(batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).uniform_(0,1)
    alpha = alpha.to(device)
    interpolated = alpha * real_images + ((1 - alpha) * fake_images)
    interpolated = interpolated.to(device)
    interpolated = Variable(interpolated, requires_grad=True)
    prob_interpolated = netD(interpolated)
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]
    lambda_term = 10
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = TrainDatasetFromFolder('../../../CelebA-HQ-img/', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('../../../CelebA-HQ-img/', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=opt.testBatchSize, shuffle=False)

    netG = Generator(UPSCALE_FACTOR).to(device)
    netD = Discriminator().to(device)

    # pretrain the Generator and load it
    # netG.load_state_dict(torch.load('epochs/' + opt.pretrain_path))

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    criterionG = Generator_loss().to(device)

    # loop over the dataset multiple times
    for epoch in range(opt.start, NUM_EPOCHS+1):
        d_total_loss = 0.0
        g_total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)


            ## update D network
            optimizerD.zero_grad()
            fake_image = netG(inputs)
            D_fake = netD(fake_image)
            D_real = netD(labels)

            d_fake_loss = torch.mean(D_fake)
            d_real_loss = torch.mean(D_real)
            gradient_penalty = calculate_gradient_penalty(labels.size(0), netD, labels, fake_image)

            d_loss = d_fake_loss - d_real_loss + gradient_penalty
            d_total_loss += d_loss
            d_loss.backward()
            optimizerD.step()

            ## update G
            # zero the parameter gradients
            optimizerG.zero_grad()
            # forward + backward + optimize
            outputs = netG(inputs)
            D_fake = netD(outputs)
            g_loss = criterionG(D_fake, outputs, labels)
            g_total_loss += g_loss

            g_loss.backward()
            optimizerG.step()

            print("===> Epoch[{}]({}/{}): gp: {:.4f}|D_real: {:.4f}|D_fake: {:.4f}|D_loss :{:.4f}|G_loss: {:.4f}".format(epoch, i, len(train_loader), gradient_penalty, d_real_loss.item(), d_fake_loss.item(), d_loss.item(), g_loss.item()))
        print("===> Epoch {} Complete: Avg. D_loss: {:.4f} G_loss: {: 4f}".format(epoch, d_total_loss / len(train_loader), g_total_loss / len(train_loader)))
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netG.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        val_loss = nn.MSELoss()
        avg_psnr = 0.0
        with torch.no_grad():
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_loader:
                sr = netG(val_lr.to(device))
                mse = val_loss(sr, val_hr.to(device))
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0)),
                     display_transform()(val_hr.squeeze(0))
                    ]
                )
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            index = 1
            for image in val_images:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1



