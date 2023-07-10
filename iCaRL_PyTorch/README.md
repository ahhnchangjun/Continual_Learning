# iCaRL PyTorch implementation

이 코드는 CIFAR-100 실험을 재현할 수 있는 완전한 iCaRL 코드 입니다.

이 코드는 현재 CIFAR-100의 10개 클래스를 한번에 처리하는 실험을 재현합니다.
(Figure 2. a)

이 코드는 공식 iCaRL 코드가 아니며, CIFAR-100 실험을 위한 원본 Theano-Lasagne 코드는 공식 레포지토리에서 찾을 수 있습니다.
[official repo](https://github.com/srebuffi/iCaRL).


tested with Python 3.8.6 and PyTorch 1.7.1 
(see the [conda environment](environment.yml) for more details).


## About this implementation

이 구현은 Theano-Lasagne 코드를 수작업으로 번역한 것입니다.

수정된 ResNet과 사용자 정의 가중치 초기화 절차가 포함됩니다.

원본 코드와 불일치가 존재 할 수 있습니다.

더 효율적이고 일반적으로 검증된 버전은 다음 링크에서 확인 할 수 있습니다.
[Avalanche](https://github.com/ContinualAI/avalanche) Continual 




## Reading the code

main_icarl.py 파일로 실험을 시작합니다.

NCProtocol 클래스는 CIFAR-100 데이터셋을 분할하여 클래스를 배치 반환합니다.

