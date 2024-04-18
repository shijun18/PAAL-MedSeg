import sys
sys.path.append('..')
from model.predictor import ap18
from model.unet import unet
from model.unet import unet
from model.utils import count_params_and_macs

if __name__ == '__main__':

    from torchsummary import summary
    import torch
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_shape = (1,1,512,512)
    n,c,h,w = input_shape

    # net = ap18(input_channels=6, num_classes=5)
    net = unet('unet',encoder_name='resnet50_dropout', in_channels=1, classes=4, aux_losspredictor=True)
      
    summary(net.cuda(),input_size=(c,h,w),batch_size=1,device='cuda')
    
    net = net.cuda()
    net.detach_flag = False
    net.train()
    input = torch.randn(input_shape).cuda()
    output = net(input)
    if isinstance(output, tuple):
        print(output[0].size())
    
    count_params_and_macs(net.cuda(),input_shape)