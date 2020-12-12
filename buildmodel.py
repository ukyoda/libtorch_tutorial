from pymodel.model import get_model
import argparse 
import torch

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('save', help='出力ファイルパス')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'mobilenet_v2'], help='変換するモデル')
    return parser.parse_args()

def main():
    args = getargs()
    model = get_model(args.model)
    model = model.eval()
    model = model.cuda()
    
    dummy = torch.zeros([1, 3, 32, 32], dtype=torch.float32).cuda()
    pt_model = torch.jit.trace(model, dummy)
    pt_model.save(args.save)

if __name__ == '__main__':
    main()