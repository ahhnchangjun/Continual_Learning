from typing import Union, Sequence, Callable

import torch
from torch.nn import Module, Sequential, BatchNorm2d, Conv2d, ReLU, ConstantPad3d, Identity, AdaptiveAvgPool2d, Linear
from torch import Tensor
from torch.nn.init import zeros_, kaiming_normal_
from torch.nn.modules.flatten import Flatten
import torch.nn.functional as F


class IdentityShortcut(Module):
    def __init__(self, transform_function: Callable[[Tensor], Tensor]):
        super(IdentityShortcut, self).__init__()
        self.transform_function = transform_function

    def forward(self, x: Tensor) -> Tensor:
        return self.transform_function(x)


def conv3x3(in_planes: int, out_planes: int, stride: Union[int, Sequence[int]] = 1):
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def batch_norm(num_channels: int) -> BatchNorm2d:
    # Default gamma and beta values are 1 and 0 in both lasagne and pytorch
    # Same for epsilon, which is 1e-4 in both frameworks
    # Same for "alpha"|"momentum" (lasagne|pytorch)
    # Both frameworks update running averages during training

    return BatchNorm2d(num_channels)


class ResidualBlock(Module):

    def __init__(self, input_num_filters: int, increase_dim: bool = False, projection: bool = False,
                 last: bool = False):
        super().__init__()
        self.last: bool = last # 마지막 블록 여부를 저장하는 불리언 변수를 초기화한다.

        if increase_dim: #차원이 증가하는 경우, 첫 번째 컨볼루션 레이어의 stride와 출력 필터 수를 서정
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:# 차원이 증가하지 않는 경우, stride (1,1)로 설정 후, 출력 필터 수를 입력 필터 수와 동일하게 설정
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        self.direct = Sequential(
            conv3x3(input_num_filters, out_num_filters, stride=first_stride),
            batch_norm(out_num_filters),
            ReLU(True),
            conv3x3(out_num_filters, out_num_filters, stride=(1, 1)),
            batch_norm(out_num_filters),
        ) # Resodial block의 직접 경로(direct path)를 정의하는 Sequential 모듈. 
        # 이 경우로는 두 개의 컨볼루션 레이어와 BN레이어로 구성되어 있으며,
        # 첫 번째 컨볼루션 레이어는 입력 필터수에서 출력 필터수로 변환하고 ReLU활성화 함수를 적용
        # 두 번째 컨볼루션 레이어는 출력 필터수를 유지하고, 다시 배치 정규화와 ReLU를 적용

        self.shortcut: Module
        # shortcut 연결을 담당하는 모듈이다. shortcut은 입력을 직접 전달하는 것으로 identity shortcut 또는 proejction shortcut 두가지 옵션 중 하나를 선택할 수있다.

        # add shortcut connections
        if increase_dim: # 차원이 증가하는 경우, projection을 선택
        # 이 경우 1x1 컨볼류션 레이어와 배치 정규화를 사용하여 입력 필터 수에서 출력 필터 수로 변환
            if projection:
                # projection shortcut, as option B in paper
                self.shortcut = Sequential(
                    Conv2d(input_num_filters, out_num_filters, kernel_size=(1, 1), stride=(2, 2), bias=False),
                    batch_norm(out_num_filters)
                )
            else: # 차원이 증가하지 않는 경우, identity shortcut을 선택
                # identity shortcut, as option A in paper
                self.shortcut = Sequential(
                    IdentityShortcut(lambda x: x[:, :, ::2, ::2]),
                    ConstantPad3d((0, 0, 0, 0, out_num_filters // 4, out_num_filters // 4), 0.0)
                )
        else:
            self.shortcut = Identity()

    def forward(self, x):
      # 입력을 받아, Residual Block을 통과시키고 출력을 반환한다. 마지막 블록인 경우, 직접 경로의 출력과 shortcut의 출력을 더하여 반환한다.
      # 그러지 않은 경우 직접 경로의 출력과 shortcut의 출력을 더한 후 ReLU활성화 함수를 적용하여 반환한다.
        if self.last:
            return self.direct(x) + self.shortcut(x)
        else:
            return torch.relu(self.direct(x) + self.shortcut(x))


class IcarlNet(Module): # iCaRL알고리즘을 위한 네트워크 achitecture을 정의한 클래스이며, Residual Block을 사용하여 feature extractor, classifer을 구성한다.
    def __init__(self, num_classes: int, n=5):
        super().__init__()

        input_dims = 3 # 입력 이미지의 채널 수
        output_dims = 16 # 첫 번째 Convolutional Layer의 출력 채널 수를 16으로 설정

        first_conv = Sequential(
            conv3x3(input_dims, output_dims, stride=(1, 1)),
            batch_norm(16),
            ReLU(True)
        )

        input_dims = output_dims
        output_dims = 16

        # first stack of residual blocks, output is 16 x 32 x 32
        layers_list = []
        for _ in range(n):
            layers_list.append(ResidualBlock(input_dims))
        first_block = Sequential(*layers_list)

        input_dims = output_dims
        output_dims = 32

        # second stack of residual blocks, output is 32 x 16 x 16
        layers_list = [ResidualBlock(input_dims, increase_dim=True)]
        for _ in range(1, n):
            layers_list.append(ResidualBlock(output_dims))
        second_block = Sequential(*layers_list)

        input_dims = output_dims
        output_dims = 64

        # third stack of residual blocks, output is 64 x 8 x 8
        layers_list = [ResidualBlock(input_dims, increase_dim=True)]
        for _ in range(1, n-1):
            layers_list.append(ResidualBlock(output_dims))
        layers_list.append(ResidualBlock(output_dims, last=True))
        third_block = Sequential(*layers_list)
        final_pool = AdaptiveAvgPool2d(output_size=(1, 1))

        self.feature_extractor = Sequential(
            first_conv, first_block, second_block, third_block, final_pool, Flatten())
        # feature extractor
        input_dims = output_dims
        output_dims = num_classes

        self.fc = Linear(input_dims, output_dims)
        # 선형 레이어를 정의하여 fc로 설정. 입력 크기는 input_dims, 출력 크기는 output_dims 이다.

    def forward(self, x):
      # 모델의 forward 연산을 정의하는 함수이며, 입력 x를 받아 출력을 계산한다.
        x = self.feature_extractor(x)  # Already flattened
        # 입력 x를 feature_extractor에 통과시켜 특성을 추출합니다.
        # 이미지 차원이 1x1 이 되기 때문에 Flatten이 적용되어 이미지를 벡터로 변환한다.
        x = self.fc(x)
        # 특성 벡터 x를 fc 레이어에 통과시켜 클래스에 대한 확률을 출력한다. 마지막 layer에는 sigmoid 함수가 적용된다.
        return torch.sigmoid(x)


def make_icarl_net(num_classes: int, n=5) -> IcarlNet:
    return IcarlNet(num_classes, n=n)


def initialize_icarl_net(m: Module):
    if isinstance(m, Conv2d):
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            zeros_(m.bias.data)

    elif isinstance(m, Linear):
        # Note: nonlinearity='sigmoid' -> gain = 1.0 as of PyTorch code. See torch.nn.init.calculate_gain(...)
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='sigmoid')
        if m.bias is not None:
            zeros_(m.bias.data)



