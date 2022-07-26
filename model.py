import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=3, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            #print(x.shape)
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class conv_deconv(nn.Module):

    def __init__(self):
        super(conv_deconv,self).__init__()
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16, kernel_size=4,stride=1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight) #Xaviers Initialisation
        self.swish1= nn.ReLU()

        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        nn.init.xavier_uniform(self.conv2.weight)
        self.swish2 = nn.ReLU()

        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        nn.init.xavier_uniform(self.conv3.weight)
        self.swish3 = nn.ReLU()

        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3)
        nn.init.xavier_uniform(self.deconv1.weight)
        self.swish4=nn.ReLU()

        #Max UnPool 1
        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5)
        nn.init.xavier_uniform(self.deconv2.weight)
        self.swish5=nn.ReLU()

        #Max UnPool 2
        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=4)
        nn.init.xavier_uniform(self.deconv3.weight)
        self.swish6=nn.ReLU()

    def forward(self,x):
        out=self.conv1(x)
        out=self.swish1(out)
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        out=self.conv2(out)
        out=self.swish2(out)
        size2 = out.size()
        out,indices2=self.maxpool2(out)
        out=self.conv3(out)
        out=self.swish3(out)

        out=self.deconv1(out)
        out=self.swish4(out)
        out=self.maxunpool1(out,indices2,size2)
        out=self.deconv2(out)
        out=self.swish5(out)
        out=self.maxunpool2(out,indices1,size1)
        out=self.deconv3(out)
        out=self.swish6(out)
        return(out)









# def test():
#     x = torch.randn((1, 1, 161, 161))
#     model = UNET(in_channels=1, out_channels=3)
#     preds = model(x)
#     print(preds)
#     #assert preds.shape == x.shape
#     print(torch.max(preds,1))
#     #print(max(preds))

# if __name__=="__main__":
#     test()

