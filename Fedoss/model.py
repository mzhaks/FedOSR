import torch
import torch.nn as nn
import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
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

class ResNet(nn.Module):
    def __init__(self, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        resnet18 = models.resnet18(pretrained=False)
        
        self.backbone = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3
        )
        
        self.layer4 = resnet18.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.main_cls = nn.Linear(512, num_classes + 1)
        
        self.auxiliary_layer4 = self._make_auxiliary_layer(BasicBlock, 256, 512, 2, stride=2)      
        self.auxiliary_cls = nn.Linear(512, num_classes)

        self._initialize_weights(zero_init_residual)

    def _initialize_weights(self, zero_init_residual):
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

    def _make_auxiliary_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    
    def aux_forward(self, x):
        x = self.auxiliary_layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        aux_out = self.auxiliary_cls(x)              
        return {'aux_out': aux_out}
    
    def discrete_forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)        
        outputs = self.main_cls(x)              
        return {'outputs': outputs}

    def forward(self, x):
        x = self.backbone(x)
        aux_out = self.aux_forward(x.clone().detach())
        
        boundary_feats = x
        discrete_feats = x
        
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        outputs = self.main_cls(x)
        
        return {
            'outputs': outputs, 
            'aux_out': aux_out['aux_out'], 
            'boundary_feats': boundary_feats, 
            'discrete_feats': discrete_feats
        }
