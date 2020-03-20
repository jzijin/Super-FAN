import argparse
import time
import math
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import pytorch_ssim
# from model import Generator
from model.model import Generator
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, RandomRotation

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_1.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
image = image.resize((512, 512))
origin = image
origin = Variable(ToTensor()(origin), volatile=True).unsqueeze(0)

x, y = image.size
image = image.resize((x//opt.upscale_factor, y//opt.upscale_factor), Image.BICUBIC)
tmp = image.resize((x, y), Image.BICUBIC)
tmp.save("BICUBIC.jpg")
# image = RandomRotation(30)(image)
# image.save("rotate.jpg")
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()
    origin = origin.cuda()

start = time.clock()
with torch.no_grad():
    out = model(image)

loss = nn.MSELoss()
mse = loss(out, origin)
psnr = 20 * math.log10(1 / math.sqrt(mse.item()))
print("psnr: ", psnr)
print("ssim:", pytorch_ssim.ssim(out,origin).item())
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
