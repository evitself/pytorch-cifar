from collections import OrderedDict
import torch
import torchvision
import torch.onnx
import argparse
from models import *

parser = argparse.ArgumentParser(description='Export model')
parser.add_argument('--input',  '-i', default=None, type=str, help='export input file')
parser.add_argument('--output', '-o', default='output.onnx', type=str, help='export output file')
args = parser.parse_args()


class InferenceModel(nn.Module):
    def __init__(self, model) -> None:
        super(InferenceModel, self).__init__()
        self.backbone = model
        self.softmax = nn.Softmax(-1)
    
    def forward(self, x):
        out = self.backbone(x)
        out = self.softmax(out)
        return out


def export_onnx(torch_weights_file, output_file):
    checkpoint = torch.load(torch_weights_file, map_location=torch.device('cpu'))
    net = PreActResNet18(num_classes=2)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        if k.startswith('module.'):
            # remove prefix if original model is trained on gpu with DataParrellel
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    # append softmax in inference model
    model = InferenceModel(net)
    x = torch.randn(9, 3, 128, 128, requires_grad=False)
    out = model(x)
    torch.onnx.export(model,
                      x,
                      output_file,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})

if __name__ == '__main__':
    export_onnx(args.input, args.output)
