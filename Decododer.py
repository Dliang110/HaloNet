import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

## Decoder ##

class decoder2(nn.Module):
    def __init__(self, in_channels):
        super(decoder2, self).__init__()
        self.decoder2 = nn.Sequential(
             nn.ConvTranspose2d(in_channels= in_channels, out_channels=1408, kernel_size= 3, stride=2,padding=1),  # In b, 8, 8, 8 >> out b, 16, 15, 15
             nn.BatchNorm2d(1408, affine = True),
             nn.ReLU(True),

             nn.ConvTranspose2d(1408, 704, 3, stride=2, padding = 1),  #out> b,32, 49, 49
             nn.BatchNorm2d(704, affine = True),
             nn.ReLU(True),

             nn.ConvTranspose2d(704, 352, 3, stride=2, padding=2),  #out> b, 32, 245, 245
             nn.BatchNorm2d(352, affine = True),
             nn.ReLU(True),

             nn.ConvTranspose2d(352, 64, 6, stride=1, padding =1),  #out> b, 16, 497, 497
             nn.BatchNorm2d(64, affine = True),
             nn.ReLU(True),

             nn.ConvTranspose2d(64, 16, 3, stride=1),  #out> b, 8, 502, 502
             nn.BatchNorm2d(16, affine = True),
             nn.ReLU(True),

             nn.ConvTranspose2d(16, 8, 6, stride=1, padding = 1),  #out> b, 3, 512, 512
             nn.BatchNorm2d(8, affine = True),
              nn.ReLU(True),

             nn.ConvTranspose2d(8, 3, 4, stride=1, padding = 1),  #out> b, 3, 512, 512
             nn.BatchNorm2d(3, affine = True),
             # nn.ConvTranspose2d(8, 3, 3, stride=1),  #out> b, 3, 512, 512
             nn.Tanh()
             )

    def forward(self, x):
         recon = self.decoder2(x)
         return recon
if __name__=="__main__":
    from torchsummary import summary
#    mod = Res18().cuda()
#    summary(mod, input_size=(3,64,64))
#
    decod = decoder2(in_channels=2816).cuda()
    summary(decod, ( 2816,64,64))
