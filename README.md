# U-Net
본 논문은 의료영상학회 중 Top conference인 MICCAI에서 발표된 논문으로 U자형으로 생긴 네트워크인 U-Net 아키텍처를 제안한다. U-Net architecture는 먼저 해상도가 줄어들었다가 다시 증가하는 형태를 갖는다. 일반적인 classification 네트워크에서는 해상도가 계속 줄어들어서 마지막에는 class의 수와 dimension의 수가 같도록 만드는 것이 일반적인데, U-Net과 같은 Semantic Segmentation같은 경우 단순히 classification결과를 구하는 것이 아니라 입력 이미지와 같은 해상도를 갖는 출력 결과를 내야하기 때문에 이러한 구조가 나온다. 해상도를 줄이는 부분을 수축 경로(Contracting path)라고 하고 이는 이미지에 존재하는 넓은 문맥(context) 정보를 처리한다. 또한 해상도를 높이는 부분을 확장 경로(Expanding path)라고 하며 정밀한 지역화(precise localization)가 가능하도록 한다. 

## Strided & Transposed Convolution

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1818718b-3ea6-4aa9-9c09-ae01d04a1ee9" height="25%" width="25%"> 　　　　　　　　<img src="https://github.com/em-1001/AI/assets/80628552/65d6c4a4-219d-438b-9adb-0e835ba23d63" height="25%" width="25%"></p>

왼쪽은 일반적으로 사용하는 Down-sampling 목적의 Strided Convolution연산으로 너비와 높이가 감소하게 된다. 반면 오른쪽과 같이 Up-sampling 목적으로는 Transposed Convolution연산이 쓰이며 Up-sampling할 input 주위에 공백을 추가하여 Up-sampling을 하고 여기서 공백은 padding이 아니라 stride에 따라 결정된다. (별다른 조건이 없으면 kernelsize -1 만큼 추가한다.) 또한 만약 Transposed Convolution시 stride가 있다면 오른쪽 이미지와 같이 input의 원소 사이에 공백을 추가한다. 결과적으로 이러한 과정을 반복해서 32x32까지 크기를 줄여준다. 

Transposed Convolutio의 output size는 input size를 $I$, stride를 $S$, kernel size를 $K$, padding size를 $P$라 했을 때 다음과 같이 구해진다. 

$$O = (I-1) \times S + K - 2P$$


## Architecture
<p align="center"><img src="https://github.com/em-1001/U-Net/assets/80628552/ae0b24aa-c7a2-436f-84ab-2498c21fb994" height="70%" width="70%"></p>

#### Contracting path
전체적인 architecture는 위와 같이 생겼다. Contracting path부터 살펴보면 572x572와 같은 수는 해상도를 의미하며 입력이 흑백이면 채널 size는 위 architecture의 input처럼 1이 된다. 이러한 input에 kernel size가 64인 Convolution Layer를 사용해서 570x570x64 출력 tensor를 얻는다. 이러한 Convolution Layer을 한 번 더 사용해서 568x568로 줄여주었고, 다음으로 max pooling을 이용해서 너비와 높이를 절반 씩으로 줄여주었다. 다음으로 다시 Convolution Layer를 사용하여 channel size는 증가시키고, 너비와 높이는 줄여준다.
conv연산에서는 일반적인 classification 연산처럼 Conv -> ReLU -> Max Pooling의 연산을 거친다. 

#### Expanding path
Expanding path에서는 반대로 up-conv를 사용해서 해상도를 증가시키고 channel size는 줄여야 하므로 Convolution Layer에서의 kernel size를 줄여준다. 이러한 과정을 반복해서 최종적으로 388x388x2로 class의 수가 2개인 output이 만들어진다. Expanding path에서 중요한 점은 Contracting path에서 사용된 feature map을 그대로 전달해서 Expanding path에서 사용할 수 있도록 한다는 것이다. Expanding path에 보이는 하얀색 tensor가 해당 부분이고 Contracting path에서 추출한 feature map을 사용할 수 있기 때문에 보다 성능이 좋아지게 된다. 

추가적으로 U-Net은 Segmentation을 위한 네크워크이기 때문에 별도의 FC Layer가 필요하지 않고, Fully Convolutional Network(FCN)으로 구성된다. 또한 Contracting path의 경우 일반적인 classification model과 동일하기 때문에 이 부분은 사전에 잘 학습되어 있는 classification model을 Encoder 형태로 사용하는 경우가 많다. 


## Overlap-tile
<p align="center"><img src="https://github.com/em-1001/U-Net/assets/80628552/50229385-c271-4e28-ac01-6ae8487f4df3" height="60%" width="60%"></p>

U-Net은 Overlap-tile 전략을 사용하는데, 이는 U-Net 구조의 특성상 출력 이미지의 해상도가 입력 이미지보다 작기 때문에 의도적으로 입력을 더욱 크게 넣는 것이다. 예를 들어 위 사진에서 노란색 영역만큼 Segmentation결과가 필요하다고 하면 그보다 더 큰 파란색 영역을 입력으로 넣는다. 이렇게 하게 되면 파란색 영역 tile과 그 오른쪽 tile이 서로 겹칠 수 밖에 없게 되고, 이미지의 왼쪽 위 부분 같은 경우(위 사진에서는 노란색 부분 왼쪽 위) 입력 이미지에는 존재하지 않는 부분이므로 미러링(mirroring)을 통해 이미지 패치를 만들어주어 네트워크에 입력으로 넣게 된다. 

## Objective Function 
U-Net은 Segmentation을 위한 네트워크이므로 다음과 같이 필셀 단위(pixel-wise) softmax를 사용한다. 

$$P_k(x) = \frac{\exp(a_k(x))}{\displaystyle\sum_{k^{'}=1}^{K} exp(a_{k^{'}}(x))}$$

$x \in \Omega$ : pixel position ($\Omega \subset Z^2$)  
$k$ : $k$ th feature channel(class)  
$a_k(x)$ : activation value of the $x$ position of the $k$ th channel

$a$를 네트워크의 출력(activation)이라고 보면 일반적인 softmax와 같은 형태이다. 다만 각각의 pixel마다 확률 값을 예측해야 하므로 모든 각 pixel $x$ 마다 확률을 구하는 형태로 softmax가 쓰인다.  

$$E = \sum_{x \in \Omega} w(x) \log \left(p_{l(X)}(x) \right)$$

$l(x)$ : true label of image $x$

학습을 위한 Loss로는 Cross Entropy를 사용한다. $l(x)$가 이미지 $x$의 true label이므로 $p_{l(X)}$ 는 true label에 대한 확률 값이다. 여기에 $\log$를 씌워서 그 확률 값이 증가할 수 있도록 학습을 진행하고, 앞에 $w(x)$ 는 추가적인 가중치 함수로 이는 각각의 pixel마다 가중치를 부여하여 더 학습이 잘 수행되거나 혹은 덜 수행되도록 조정한다. 

$$w(x) = w_c(x) + w_0 \cdot \exp\left(-\frac{(d_1(x)+d_2(x))^2}{2\sigma^2} \right)$$

$w_c$ : $\Omega \to R$ : The weight map to balance the class frequencies.    
$d_1$ : $\Omega \to R$ : The distance to the border of the nearest cell.    
$d_2$ : $\Omega \to R$ : The distance to the border of the second nearest cell.  

$w(x)$는 위와 같이 계산되며 세포(cell)을 명확하게 구분하기 위해 작은 분리 경계(small separation border)를 학습한다. 

<p align="center"><img src="https://github.com/em-1001/U-Net/assets/80628552/28fde188-979e-4f9d-9c53-8693aad3d226" height="75%" width="75%"></p>

세포(cell)가 **b** 사진에서 보이는 것과 같이 붙어있을 수도 있기 때문에 이를 명확하게 구분하기 위해서 가중치를 사용하는 것이다. $w_c$는 각각의 class마다 나타나는 빈도가 다를 수 있기 때문에 이를 조율하기 위한 가중치이고, $d_1$은 가장 가까운 세포 경계까지의 거리를 의미하며 $d_2$는 두 번째로 가까운 세포 경계까지의 거리를 의미한다. 따라서 가중치 함수의 두 번째 term을 살펴보면 지수함수에 마이너스가 붙어서 거리값이 들어가므로 거리값이 작으면 작을 수록 가중치가 커지고 거리가 크면 가중치가 작아지게 된다. 즉, 인접한 세포(touching cell)사이에 있는 배경 레이블에 대해 높은 가중치를 부여하여 명확하게 분리가 되도록 한다. 
사진 **c** 를 보면 인접한 세포에 검은색 배경선이 있어 명확하게 배경으로 분류된 것을 확인할 수 있다. 


## Data Augmentation
본 논문에서는 레이블 정보가 있는(annotated) 즉, Ground truth가 있는 데이터가 적은 상황에서 효율적인 데이터 증진(data augmentation)기법을 제안한다. 의료 영상의 경우 이미지에 label을 넣기 위해서는 높은 수준의 전문의들이 직접 label을 넣어야 하기 때문에 비용이 매우 크므로 이러한 data augmentation 기법은 매우 중요하다. 

<p align="center"><img src="https://github.com/em-1001/U-Net/assets/80628552/a26bf9f6-6452-4c10-b280-27a12bbff833" height="45%" width="45%"></p>

본 논문에서는 일반적인 data augmentation에 더해 추가적으로 위 사진과 같이  Elastic Deformation 기법을 사용하였다. ElasticTransform은 각각 grid에 대해 보다 비선형적으로 변형을 가해 학습 데이터로 사용하는 방식이다. 

또한 image segmentation분야는 입력 데이터와 출력 데이터가 모두 이미지 형태이기 때문에 이러한 data augmentation을 사용할 때 입력 이미지와 출력 이미지가 되는 mask image에 대해서도 같은 transformation을 적용하는 것이 일반적이다. 

## Experiments 
<p align="center"><img src="https://github.com/em-1001/U-Net/assets/80628552/d7015dc8-cf08-4b1c-a61a-6d09adeece7a" height="75%" width="75%"></p>

**PHC-U373** : 35개의 부분적으로 주석이 있는(annotated) 학습 이미지 데이터 세트  
**DIC-HeLa** : 20개의 부분적으로 주석이 있는(annotated) 학습 이미지 데이터 세트

위 두 개의 데이터 세트로 실험을 진행해본 결과 위 사진 처럼 segmentation을 잘 수행하는 것을 확인할 수 있다. 
**b** 사진의 청록색은 모델이 예측한 부분이고, 노란색 테두리가 ground truth인데 결과가 거의 유사한 것을 확인할 수 있다. **d** 사진도 마찬가지로 모델의 예측 결과가 ground truth가 유사함을 확인할 수 있다. 


# Reference
## Web Link
https://www.youtube.com/watch?v=n_FDGMr4MxE&feature=mweb_3_dot_11268432&itc_campaign=mweb_3_dot_11268432&redirect_app_store_ios=1&app=desktop  


## Paper
U-Net : https://arxiv.org/pdf/1505.04597.pdf  
Elastic Deformations for Data Augmentation in Breast Cancer Mass Detection : https://web.fe.up.pt/~jsc/publications/conferences/2018EMecaBHI.pdf
