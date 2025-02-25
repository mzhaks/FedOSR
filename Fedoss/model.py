import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=48, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 48:
            raise ValueError('BasicBlock only supports groups=1 and base_width=48')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=48, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 48
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be a 3-element tuple")

        self.groups = groups
        self.base_width = width_per_group

        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Main Feature Extractor Layers
        self.layer1 = self._make_layer(block, 48, layers[0])
        self.layer2 = self._make_layer(block, 96, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 192, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 384, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # Classification Heads
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.main_cls = nn.Linear(384, num_classes+1)

        # Auxiliary Layer for Boundary Features
        self.auxiliary_layer4 = self._make_auxiliary_layer(block, 192, 384, layers[3], stride=2,
                                                           dilate=replace_stride_with_dilation[2])
        self.auxiliary_cls = nn.Linear(384, num_classes)

        # Initialize Weights
        self._initialize_weights(zero_init_residual)

    def _initialize_weights(self, zero_init_residual):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Creates a ResNet layer"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer) for _ in range(1, blocks))

        return nn.Sequential(*layers)

    def _make_auxiliary_layer(self, block, inplanes, planes, blocks, stride=1, dilate=False):
        """Creates an auxiliary layer"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        self.inplanes = inplanes  # Reset inplanes for auxiliary branch

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer) for _ in range(1, blocks))

        return nn.Sequential(*layers)

    def aux_forward(self, x):
        """Forward pass through the auxiliary branch"""
        x = self.auxiliary_layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        aux_out = self.auxiliary_cls(x)
        return {'aux_out': aux_out}

    def discrete_forward(self, x):
        """Forward pass through the main classifier"""
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        outputs = self.main_cls(x)
        return {'outputs': outputs}

    def forward(self, x):
        """Full forward pass with boundary and discrete features"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Extract auxiliary boundary features
        aux_out = self.aux_forward(x.detach())

        boundary_feats = x
        discrete_feats = x

        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        outputs = self.main_cls(x)

        return {'outputs': outputs, 'aux_out': aux_out['aux_out'], 
                'boundary_feats': boundary_feats, 'discrete_feats': discrete_feats}




def _resnet(arch, pretrained_path=None, **kwargs):
    model = ResNet(BasicBlock,[2,2,2,2],**kwargs) 

    if pretrained_path:
        try:
            pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

            model.load_state_dict(pretrained_dict, strict=False)  
            print(f"Loaded {len(pretrained_dict)} / {len(model_dict)} layers from {pretrained_path}")

        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    return model

def resnet18(pretrained_path=None, num_classes=7, **kwargs):
    return _resnet('resnet18', pretrained_path=pretrained_path, num_classes=num_classes, **kwargs)


model = resnet18()

