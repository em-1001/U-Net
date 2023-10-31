# U-Net
본 논문은 의료영상학회 중 Top conference인 MICCAI에서 발표된 논문으로 U자형으로 생긴 네트워크인 U-Net 아키텍처를 제안한다. U-Net architecture는 먼저 해상도가 줄어들었다가 다시 증가하는 형태를 갖는다. 일반적인 classification 네트워크에서는 해상도가 계속 줄어들어서 마지막에는 class의 수와 dimension의 수가 같도록 만드는 것이 일반적인데, U-Net과 같은 Semantic Segmentation같은 경우 단순히 classification결과를 구하는 것이 아니라 입력 이미지와 같은 해상도를 갖는 출력 결과를 내야하기 때문에 이러한 구조가 나온다. 해상도를 줄이는 부분을 수축 경로(Contracting path)라고 하고 이는 이미지에 존재하는 넓은 문맥(context) 정보를 처리한다. 또한 해상도를 높이는 부분을 확장 경로(Expanding path)라고 하며 정밀한 지역화(precise localization)가 가능하도록 한다. 

## Strided & Transposed Convolution

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1818718b-3ea6-4aa9-9c09-ae01d04a1ee9" height="25%" width="25%"> 　　　　　　　　<img src="https://github.com/em-1001/AI/assets/80628552/65d6c4a4-219d-438b-9adb-0e835ba23d63" height="25%" width="25%"></p>

왼쪽은 일반적으로 사용하는 Down-sampling 목적의 Strided Convolution연산으로 너비와 높이가 감소하게 된다. 반면 오른쪽과 같이 Up-sampling 목적으로는 Transposed Convolution연산이 쓰이며 Up-sampling할 input 주위에 공백을 추가하여 Up-sampling을 하고 여기서 공백은 padding이 아니라 stride에 따라 결정된다. (별다른 조건이 없으면 kernelsize -1 만큼 추가한다.) 또한 만약 Transposed Convolution시 stride가 있다면 오른쪽 이미지와 같이 input의 원소 사이에 공백을 추가한다. 

Transposed Convolutio의 output size는 input size를 $I$, stride를 $S$, kernel size를 $K$, padding size를 $P$라 했을 때 다음과 같이 구해진다. 

$$O = (I-1) \times S + K - 2P$$


## Architecture
<p align="center"><img src="https://github.com/em-1001/U-Net/assets/80628552/ae0b24aa-c7a2-436f-84ab-2498c21fb994" height="60%" width="60%"></p>



본 논문에서는 레이블 정보가 있는(annotated) 즉, Ground truth가 있는 데이터가 적은 상황에서 효율적인 데이터 증진(data augmentation)기법을 제안한다. 의료 영상의 경우 이미지에 label을 넣기 위해서는 높은 수준의 전문의들이 직접 label을 넣어야 하기 때문에 비용이 매우 크므로 이러한 data augmentation 기법은 매우 중요하다. 
