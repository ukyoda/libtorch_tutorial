import torchvision.models as models
import torch
import torch.nn as nn
from .imagenet_labels import LABELS

class ClassificationModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self._base = base
        self._softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        y = self._base(x)
        return self._softmax(y)

def get_model(model='resnet50'):
    if model == 'resnet18':
        base = models.resnet18(pretrained=True)
    elif model == 'resnet50':
        base = models.resnet50(pretrained=True)
    elif model == 'mobilenet_v2':
        base = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError('そのモデルは作成できません')
    return ClassificationModel(base)

if __name__ == '__main__':
    print('Create Model...')
    model = get_model()
    model = model.eval()
    model = model.cuda()
    print('model setup Complete!!')
    x = torch.zeros([1, 3, 32, 32], dtype=torch.float32).cuda()
    y = model(x)
    predicted_idx = torch.argmax(y[0]).cpu().item()
    print(predicted_idx)
    label = LABELS[predicted_idx]
    print(f'{label}, score={y[0, predicted_idx].cpu().item()}')