import torchvision.models as models
import torch
from .imagenet_labels import LABELS

def get_model(model='resnet18'):
    if model == 'resnet18':
        return models.resnet18(pretrained=True)
    elif model == 'mobilenet_v2':
        return models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError('そのモデルは作成できません')

if __name__ == '__main__':
    print('create model')
    model = get_model()
    print('create model complete')
    model = model.eval()
    print('model is eval mode')
    model = model.cuda()
    print('model is cuda mode')
    x = torch.zeros([1, 3, 32, 32], dtype=torch.float32).cuda()
    y = model(x)
    predicted_idx = torch.argmax(y[0]).cpu().item()
    print(predicted_idx)
    label = LABELS[predicted_idx]
    print(f'{label}, score={y[0, predicted_idx].cpu().item()}')