import torch
import torch.nn as nn 

from utils import l2_norm
from models.resnet import *

# Output dim of each ResNet
__BACKBONE_OUT_DIM = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048
}


def get_backbone_feat_size(backbone):
    if backbone not in __BACKBONE_OUT_DIM:
        raise ValueError('Backbone not supported: {}'.format(backbone))
    return __BACKBONE_OUT_DIM[backbone]


def extract_features(device, net, loader, return_labels=False):
    features = None
    labels = None
    net.eval()
    with torch.no_grad():
        for inputs in loader:
            images = inputs[0].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                f = net(images)['features']     # net forward returns a dict 
            f = l2_norm(f)
            if features is not None:       # after the first loop
                features = torch.cat((features, f), 0)
                labels = torch.cat((labels, inputs[1]), 0) if return_labels else None
            else:                       # during the first loop
                features = f
                labels = inputs[1] if return_labels else None
    if return_labels:
        return features.detach().cpu(), labels.detach().cpu()
    return features.detach().cpu().numpy()


class Incremental_ResNet(nn.Module):
    def __init__(self, 
                 num_classes=100, 
                 feat_size=99, 
                 backbone='resnet18', 
                 pretrained = False,
                ):
        
        super(Incremental_ResNet, self).__init__()
        self.feat_size = feat_size
        
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained)
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained)
        elif backbone == 'resnet101':
            self.backbone = resnet101(pretrained)
        elif backbone == 'resnet152':
            self.backbone = resnet152(pretrained)
        else:
            raise ValueError('Backbone not supported: {}'.format(backbone))

        self.out_dim = self.backbone.out_dim
        self.feat_size = feat_size
        
        self.fc1 = None 
        self.fc2 = None
        if self.out_dim != self.feat_size:
            print(f"add a linear layer from {self.out_dim} to {self.feat_size}")
            self.fc1 = nn.Linear(self.out_dim, self.feat_size, bias=False)
        self.fc2 = nn.Linear(self.feat_size, num_classes, bias=False)  # classifier
        
            
    def forward(self, x, return_dict=True):
        x = self.backbone(x)

        if self.fc1 is not None:
            z = self.fc1(x)
        else:
            z = x
        
        y = self.fc2(z)

        if return_dict:
            return {'backbone_features': x,
                    'logits': y, 
                    'features': z
                    }

        else:
            return x, y, z

# Creates an Incremental ResNet given the backbone, if pretrained, feature size and num classes
# If the model weights need to be loaded, load path need to be "WANDB_PROJECT+model_path" and wandb_run cannot be None
def create_model(backbone, pretrained, feat_size, num_classes, device, load_path=None, wandb_run = None):

    print(f"Creating model with {num_classes} classes and {feat_size} features")

    model = Incremental_ResNet(num_classes, feat_size, backbone, pretrained)

    if load_path is not None:
        print(f"Loading Weights from {load_path}")
        artifact = wandb_run.use_artifact(load_path, type='model')
        artifact_dir = artifact.download()
        model.load_state_dict(torch.load(artifact_dir+'/model.pth'))
    
    model.to(device=device)

    return model