# !pip install tensorboardX
import os
import numpy as np
import torch
import torch.nn as nn
import torchsummary
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt




from dataset import mirroring_Extrapolate
from dataset import Dataset
from transform import ToTensor, Normalization, RandomFlip
from model import UNet_ISBI
from loss import fn_loss



 # 파라미터 설정
 lr = 1e-4
 batch_size = 4
 # num_epoch =100
 num_epoch = 10

 data_dir = '/content/drive/MyDrive/UNET-tf2-main/UNET-tf2-main/isbi_2012/raw_data'
 ckpt_dir = '/content/ckeckpoint'
 log_dir = '/content/log'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 데이터로더 확인
dataset_train = Dataset(data_dir = os.path.join(data_dir, 'train'))

data = dataset_train.__getitem__(0)

input = data['input']
label = data['label']

test = torch.from_numpy(input.transpose((2, 0, 1)))
print(test.shape)

# squeeze()를 사용해 channel처리후 시각화
plt.subplot(121)
plt.imshow(input.squeeze())
plt.subplot(122)
plt.imshow(label.squeeze())
plt.show()

# 트랜스폼
transform = transforms.Compose([Normalization(mean=0.5, std=0.5),
                                RandomFlip(),
                                ToTensor()])

# dataset_train
dataset_train = Dataset(data_dir = os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 8)

dataset_val = Dataset(data_dir = os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = False, num_workers = 8)


net = UNet_ISBI().to(device)

fn_loss = nn.BCEWithLogitsLoss().to(device)

optim = torch.optim.Adam(net.parameters(), lr=lr)

num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

# output을 저장하기 위한 함수

# tensor to numpy
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

# norm to denorm
fn_denorm = lambda x, mean, std: (x * std) + mean

# network output을 binary로 분류
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


# 네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net' : net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim,


# reference : https://sd118687.tistory.com/11?category=549654
# input : 960 x 960 / output 572 x 572 4개
def divide_input(img):
    left_top = torchvision.transforms.functional.crop(img, 0, 0, 572, 572)
    right_top = torchvision.transforms.functional.crop(img, 388, 0, 572, 572)
    left_bottom = torchvision.transforms.functional.crop(img, 0, 388, 572, 572)
    right_bottom = torchvision.transforms.functional.crop(img, 388, 388, 572, 572)
    return left_top, right_top, left_bottom, right_bottom

# input : 960 x 960 / output 388 x 388 4개
def divide_label(img):
    left_top = torchvision.transforms.functional.crop(img, 92, 92, 388, 388)
    right_top = torchvision.transforms.functional.crop(img, 480, 92, 388, 388)
    left_bottom = torchvision.transforms.functional.crop(img, 92, 480, 388, 388)
    right_bottom = torchvision.transforms.functional.crop(img, 480, 480, 388, 388)
    return left_top, right_top, left_bottom, right_bottom


# Training
net, optim, st_epoch = load(ckpt_dir = ckpt_dir, net=net, optim=optim)


st_epoch = 0
for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []
    print('epoch : ', epoch)

    for batch, data in enumerate(loader_train, 1):
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)


        input_0, input_1, input_2, input_3 = divide_input(input)
        label_0, label_1, label_2, label_3 = divide_label(label)

        input_list = []
        label_list = []

        input_list.append(input_0)
        input_list.append(input_1)
        input_list.append(input_2)
        input_list.append(input_3)

        label_list.append(label_0)
        label_list.append(label_1)
        label_list.append(label_2)
        label_list.append(label_3)

        if(epoch==10):
            for c in range(4):
                # https://discuss.pytorch.org/t/plot-4d-tensor-as-image/147566/2
                plt.subplot(1,4,c+1)
                label_img = label_list[c].cpu()[0]
                label_img = label_img.permute(1, 2, 0).numpy()
                plt.imshow(label_img, cmap ='gray')
                plt.axis('off')
            plt.show()

        for i in range(4):
            # print('input shape : ', input_list[i].shape) -> input shape :  torch.Size([4, 1, 572, 572]) which is [batch_size, channels, height, width]
            # print('label shape : ', label_list[i].shape) -> label shape :  torch.Size([4, 1, 388, 388])
            output = net(input_list[i])

            if(epoch==10):
                plt.subplot(1,4,i+1)
                output_img = output.cpu()[0]
                output_img = output_img.permute(1, 2, 0).detach().numpy()
                plt.imshow(output_img, cmap ='RdPu')
                plt.axis('off')


            # print('output shape : ', output.shape) -> output shape :  torch.Size([4, 1, 388, 388])

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label_list[i])
            loss.backward()
            optim.step()


            # Loss
            loss_arr += [loss.item()]
            print("Train : Epoch %04d / %04d | Batch %04d / %04d | Loss %.4f"% (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))



            # 텐서보드에 저장
            #label = fn_tonumpy(label)
            #input = fn_tonumpy(fn_denorm(input, mean = 0.5, std = 0.5))
            #output = fn_tonumpy(fn_class(output))

            #writer_train.add_image('label', label, num_batch_train * (epoch -1) + batch, dataformats ='NHWC')
            #writer_train.add_image('input', input, num_batch_train * (epoch -1) + batch, dataformats ='NHWC')
            #writer_train.add_image('output', output, num_batch_train * (epoch -1) + batch, dataformats ='NHWC')
        #writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
        plt.show()

    # validation
    with torch.no_grad():
        net.eval()
        loss_arr = []
        for batch, data in enumerate(loader_val, 1):

            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            input_0, input_1, input_2, input_3 = divide_input(input)
            label_0, label_1, label_2, label_3 = divide_label(label)

            input_list = []
            label_list = []

            input_list.append(input_0)
            input_list.append(input_1)
            input_list.append(input_2)
            input_list.append(input_3)

            label_list.append(label_0)
            label_list.append(label_1)
            label_list.append(label_2)
            label_list.append(label_3)


            for i in range(4):
                output = net(input_list[i])

                # Loss
                loss = fn_loss(output, label_list[i])
                loss_arr += [loss.item()]
                print("Val : Epoch %04d / %04d | Batch %04d / %04d | Loss %.4f"% (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

                #label = fn_tonumpy(label)
                #input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                #output = fn_tonumpy(fn_class(output))
                #writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                #writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                #writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

    #writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
    if epoch % 5 == 0:
        save(ckpt_dir = ckpt_dir, net = net, optim= optim, epoch = epoch)
#writer_train.close()
#writer_val.close()
