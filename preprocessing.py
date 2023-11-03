import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir_data = '/content/drive/MyDrive/UNET-tf2-main/UNET-tf2-main/isbi_2012/raw_data'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

nx, ny = img_label.size
nframe = img_label.n_frames

print("nx = " ,nx)
print("ny = " ,ny)
print("nframe = ", nframe)

'''
nx =  512
ny =  512
nframe =  30
'''

nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 디렉토리 생성
if not os.path.exists(dir_save_train):
  os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
  os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
  os.makedirs(dir_save_test)


id_frame = np.arange(nframe)
np.random.shuffle(id_frame)
print(id_frame)

'''
[ 5 27 29  0 15  8  1 17  9 11  2 28 12 14  3  4 13 21 22 20 26  7 19 23
 24 10 18  6 16 25]
'''

# 셔플된 프레임 저장
offset_nframe = 0
for i in range(nframe_train):
  img_label.seek(id_frame[i + offset_nframe])
  img_input.seek(id_frame[i + offset_nframe])

  label_ = np.asarray(img_label)
  input_ = np.asarray(img_input)
  print("label_ =",label_)
  print("input_ =",input_)
  print("=" *100)
  np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
  np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)


# 셔플된 프레임 저장
offset_nframe  += nframe_val
for i in range(nframe_test):
  img_label.seek(id_frame[i + offset_nframe])
  img_input.seek(id_frame[i + offset_nframe])

  label_ = np.asarray(img_label)
  input_ = np.asarray(img_input)

  np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
  np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

plt.subplot(121)
plt.imshow(label_, cmap ='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap ='gray')
plt.title('input')
plt.show()



