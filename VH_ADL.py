# -*- coding: utf-8 -*-
"""
@author: Emilie Mathian
"""

import torch
import torch.nn as nn
import Halonet
from  Decoder  import  decoder2


#### NETWORK DECLARATION ####
# torch.autograd.set_detect_anomaly(True) # this is to check any problem in the network by backtracking

class VH_AE(nn.Module):
    def __init__(self, train = True):

        super(VH_AE, self).__init__()
        self.halonetH6 = Halonet.halonetB6( )
        self.decoder = decoder2(in_channels = 2816)
        self.Train =  train
        if self.Train:
            print("\nInitializing network weights.........")
            initialize_weights(self.halonetH6, self.decoder)

    def forward(self,x):
        b = x.size(0)
        encoded = self.halonetH6(x)
        # if self.Train:
        #     encoded = add_noise(encoded)
        # encoded1, vectors = self.Digcap(encoded.view(b,encoded.size(1)*8*8,-1))
        recons = self.decoder(encoded)
        # pi, mu, sigma = self.G_estimate(encoded)
        # return encoded, pi, sigma, mu, recons

        return encoded, recons

# Initialize weight function
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

##### Adding Noise ############

# def add_noise(latent, noise_type="gaussian", sd=0.2):
#     """Here we add noise to the latent features concatenated from the 4 autoencoders.
#     Arguements:
#     'gaussian' (string): Gaussian-distributed additive noise.
#     'speckle' (string) : Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
#     'sd' (integer) : standard deviation used for geenrating noise
#
#     Input :
#         latent : numpy array or cuda tensor.
#
#     Output:
#         Array: Noise added input, can be np array or cuda tnesor.
#     """
#     assert sd >= 0.0
#     if noise_type == "gaussian":
#         mean = 0.
#
#         n = torch.distributions.Normal(torch.tensor([mean]), torch.tensor([sd]))
#         noise = n.sample(latent.size()).squeeze(-1).cuda()
#         latent = latent + noise
#         return latent
#
#     if noise_type == "speckle":
#         noise = torch.randn(latent.size()).cuda()
#         latent = latent + latent * noise
#         return latent
#TETS
if __name__ == "__main__":
    from torchsummary import summary

    mod = VH_AE().cuda()
    print(mod)
    summary(mod, (3,256,256))


{"mode":"full","isActive":false}
