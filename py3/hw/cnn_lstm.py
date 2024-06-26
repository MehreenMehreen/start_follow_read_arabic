
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
            
        #print('blstm input', input.size())
        recurrent, notused = self.rnn(input)
        #print('rnn output', recurrent.size(), 'not used', notused)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        #print('.....', output.size())
        return output

class CRNN(nn.Module):

    def __init__(self, cnnOutSize, nc, nclass, nh, n_rnn=2, leakyRelu=False, use_instance_norm=False):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                if not use_instance_norm:
                    cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                else:
                    cnn.add_module(f'instancenorm{i}', nn.InstanceNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        # Mehreen: nclass is the total outut characters. nh is set to 512 by create_model
        self.rnn = BidirectionalLSTM(cnnOutSize, nh, nclass)
        ###MEHREEN ADD PARAM dim=2
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()   
        #print('.....', input.size())
        #print('....', b, c, h, w)
        
        if torch.any(torch.isnan(conv)):
            print("CONV IS NAN (b,c,h,w) = ", b, c, h, w)

            #iimg = input.cpu()[0].permute(2, 1, 0)
            #print('....iimg.size', input.size())
            #plt.imshow(iimg)
            #plt.show()
            
        ####MEHREEN change this    
        #conv = conv.view(b, -1, w) ###<--original
        # to
        conv = torch.reshape(conv, (b, c*h, w))
        ###End mehreen
        conv = conv.permute(2, 0, 1)  # [w, b, c]        
        # rnn features
        output = self.rnn(conv)
        if torch.any(torch.isnan(output)):
            print("OUTPUT FROM RNN IS NAN")
        ###MEHREEN ADD    
        output = self.softmax(output)    
        if torch.any(torch.isnan(output)):
            print("OUTPUT FROM SOFTMAX IS NAN")
        if torch.any(torch.isinf(output)):
            print("OUTPUT FROM SOFTMAX IS INF")    
        ###END MEHREEN
        return output

def create_model(config):
    use_instance_norm = False
    if 'use_instance_norm' in config and config['use_instance_norm']:
        use_instance_norm = True
    crnn = CRNN(config['cnn_out_size'], config['num_of_channels'], config['num_of_outputs'], 512, 
                use_instance_norm=use_instance_norm)
    return crnn

