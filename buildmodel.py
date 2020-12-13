from pymodel.model import get_model
import argparse 
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('save', help='出力ファイルパス')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50', 'mobilenet_v2'], help='変換するモデル')
    parser.add_argument('--src', help='テスト用の画像データ')
    return parser.parse_args()

def preprocess(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess(image)
    

def main():
    args = getargs()
    print('Create Model...')
    model = get_model(args.model)
    model = model.eval()
    model = model.cuda()
    print('Model Created!!')

    if args.src is not None:
        image = Image.open(args.src)
    else:
        image = None
    if image is not None:
        print('Crop Size')
        image = preprocess(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        scores = model(image)
        indices = scores.argmax(dim=1)
        index = indices[0].item()
        print(index, scores[0, index].item())

    dummy = torch.zeros([1, 3, 224, 224], dtype=torch.float32).cuda()
    pt_model = torch.jit.trace(model, dummy)
    pt_model.save(args.save)

if __name__ == '__main__':
    main()