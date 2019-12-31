import argparse
from math import log10

import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder

from model.model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# attention the crop_size must be 2^n because of the hourglass layer!!!!
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=1000, type=int, help='train epoch number')
parser.add_argument('--batchSize', default=64, type=int, help='train batchSize')
parser.add_argument('--testBatchSize', default=1, type=int, help='test batchSize')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--start', default=1, type=int, help='start num')


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


    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    # loop over the dataset multiple times
    for epoch in range(opt.start, NUM_EPOCHS+1):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizerG.zero_grad()

            # forward + backward + optimize
            outputs = netG(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizerG.step()

            running_loss += loss.item()
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, i, len(train_loader), loss.item()))
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, running_loss / len(train_loader)))
        torch.save(netG.state_dict(), 'epochs/pretrain_netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

        # eval
        avg_psnr = 0
        with torch.no_grad():
            for batch in val_loader:
                input, _, target = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                prediction = netG(input)
                mse = criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
            print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(val_loader)))

