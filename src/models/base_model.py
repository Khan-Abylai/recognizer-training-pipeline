import torch.nn as nn
import torch.nn.init as initializer
from torch.cuda.amp import autocast
import string
from torchvision.models import resnet18
import torch
class Embedding(nn.Module):

    def __init__(self, input_size, hidden_size, num_class, bidirectional=False, num_layers=1):
        super(Embedding, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional)
        self.num_class = num_class
        if bidirectional:
            hidden_size = hidden_size * 2

        self.embedding = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden_size, num_class), )

        initializer.xavier_uniform_(self.rnn.weight_hh_l0)
        initializer.xavier_uniform_(self.rnn.weight_ih_l0)
        initializer.xavier_uniform_(self.embedding[1].weight)

    @autocast()
    def forward(self, x):
        recurrent, _ = self.rnn(x)

        batch_size, sequence_size, hidden_size = recurrent.size()
        t_rec = recurrent.reshape(sequence_size * batch_size, hidden_size)
        embedding = self.embedding(t_rec)
        result = embedding.reshape(batch_size, sequence_size, self.num_class).log_softmax(2)
        return result

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.25))

    @autocast()
    def forward(self, x):
        return self.block(x)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x

class CRNN(nn.Module):

    def __init__(self, image_h, num_class, num_layers, is_lstm_bidirectional):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(

            ConvBlock(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0), )

        self.squeeze = nn.Sequential(nn.Linear(512 * 6, 256), nn.ReLU(inplace=True), nn.Dropout(0.5))

        initializer.xavier_uniform_(self.squeeze[0].weight)

        self.rnn = Embedding(256, 256, num_class, bidirectional=is_lstm_bidirectional, num_layers=num_layers)

    @autocast()
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 512 * 6, 30).permute(0, 2, 1).contiguous()
        x = self.squeeze(x)
        return self.rnn(x)

class CRNN_2(nn.Module):

    def __init__(self, image_h, num_class, num_layers, is_lstm_bidirectional, num_regions):
        super(CRNN_2, self).__init__()

        self.cnn = nn.Sequential(

            ConvBlock(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0), )

        self.squeeze = nn.Sequential(nn.Linear(512 * 14, 256), nn.ReLU(inplace=True), nn.Dropout(0.5))

        self.classificator = nn.Sequential(nn.Linear(9728, num_regions))

        initializer.xavier_uniform_(self.squeeze[0].weight)

        self.rnn = Embedding(256, 256, num_class, bidirectional=is_lstm_bidirectional, num_layers=num_layers)

    @autocast()
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 512 * 14, 38).permute(0, 2, 1).contiguous()
        x = self.squeeze(x)
        cls = x.view(x.size(0), -1)
        return self.rnn(x), self.classificator(cls)

class BlockRNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, bidirectional, recurrent_nn=nn.LSTM):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.bidirectional = bidirectional

        # layers
        self.rnn = recurrent_nn(in_size, hidden_size, bidirectional=bidirectional, batch_first=True)

    def forward(self, batch, add_output=False):
        """
        in array:
            batch - [seq_len , batch_size, in_size]
        out array:
            out - [seq_len , batch_size, out_size]
        """
        outputs, hidden = self.rnn(batch)
        out_size = int(outputs.size(2) / 2)
        if add_output:
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output
class CRNN_resnet(nn.Module):

    def __init__(self, image_h, image_w, num_class, hl, is_lstm_bidirectional, linear_size):
        super(CRNN_resnet, self).__init__()
        backbone = resnet18
        conv_nn = backbone(pretrained=True)
        conv_modules = list(conv_nn.children())[:-3]
        self.conv_nn = nn.Sequential(*conv_modules)
        _, backbone_c, backbone_h, backbone_w = self.conv_nn(torch.rand((1, 3, image_h, image_w))).shape


        self.linear1 = nn.Linear(backbone_c*backbone_h, linear_size)
        self.recurrent_layer1 = BlockRNN(linear_size, hl, hl,
                                         bidirectional=is_lstm_bidirectional)
        self.recurrent_layer2 = BlockRNN(hl, hl, num_class,
                                         bidirectional=is_lstm_bidirectional)

        self.linear2 = nn.Linear(hl * 2, num_class)

    @autocast()
    def forward(self, batch: torch.float64):
        """
        forward
        """
        batch_size = batch.size(0)

        # convolutions
        batch = self.conv_nn(batch)

        # make sequences of image features
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        batch = batch.reshape(batch_size, n_channels, -1)
        batch = self.linear1(batch)

        # rnn layers
        batch = self.recurrent_layer1(batch, add_output=True)
        batch = self.recurrent_layer2(batch)
        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)
        return batch

class CRNN_lite(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, lstmFlag=True):
        super(CRNN_lite, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [5, 3, 3, 3, 3, 3, 2]
        ps = [2, 1, 1, 1, 1, 1, 0]
        # ss = [1, 1, 1, 1, 1, 1, 1]
        ss = [2, 1, 1, 1, 1, 1, 1]
        # nm = [24, 128, 256, 256, 512, 512, 512]
        nm = [32, 64, 128, 128, 256, 256, 256]
        # exp_ratio = [2,2,2,2,1,1,2]
        self.lstmFlag = lstmFlag

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            # exp  = exp_ratio[i]
            # exp_num = exp * nIn
            if i == 0:
                cnn.add_module('conv_{0}'.format(i),
                               nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                cnn.add_module('relu_{0}'.format(i), nn.ReLU(True))
            else:

                cnn.add_module('conv{0}'.format(i),
                               nn.Conv2d(nIn, nIn, ks[i], ss[i], ps[i], groups=nIn))
                if batchNormalization:
                    cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nIn))
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

                cnn.add_module('convproject{0}'.format(i),
                               nn.Conv2d(nIn, nOut, 1, 1, 0))
                if batchNormalization:
                    cnn.add_module('batchnormproject{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        # cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)

        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16

        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d((2, 2))) # 256x4x16

        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16

        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2))) # 256x4x16

        convRelu(6, True)  # 512x1x16


        self.cnn = cnn
        if self.lstmFlag:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(nm[-1], nh // 2, nh),
                BidirectionalLSTM(nh, nh // 4, nclass)
            )
        # else:
        #     self.linear = nn.Sequential(
        #         nn.Linear(nm[-1], nh // 2),
        #         nn.Linear(nh // 2, nclass),
        #     )


    @autocast()
    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        stop = 1

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        if self.lstmFlag:
            # rnn features
            output = self.rnn(conv)
        # else:
        #     T, b, h = conv.size()
        #
        #     t_rec = conv.contiguous().view(T * b, h)
        #
        #     output = self.linear(t_rec)  # [T * b, nOut]
        #     output = output.view(T, b, -1)

        return output

class CRNN_3(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN_3, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1

        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)
        output = self.dense(recurrent)

        return output   # shape: (seq_len, batch, num_class)

class CRNN_3_cls(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False, num_regions=None):
        super(CRNN_3_cls, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.squeeze = nn.Sequential(nn.Linear(1536 * 1, 39))

        self.classificator = nn.Sequential(nn.Linear(1521, num_regions))

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)


    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1

        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        x = conv.view(-1, 1536 * batch, 39).permute(0, 2, 1).contiguous()
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)
        output = self.dense(recurrent)

        cls = self.classificator(x)
        return output, cls   # shape: (seq_len, batch, num_class)


if __name__ == '__main__':
    alphabet = string.digits + string.ascii_lowercase
    num_class = len(alphabet) + 1
    # net = CRNN(image_h=32, num_class=num_class, num_layers=1, is_lstm_bidirectional=False)
    # # print(net)
    # net2 = CRNN_2(image_h=64, num_class=num_class, num_layers=2, is_lstm_bidirectional=True, num_regions=7)
    # print(net2)
    # resn = CRNN_resnet(image_h=64, image_w=160, num_class=num_class, hl=32, is_lstm_bidirectional=True, linear_size=512)
    # # print(resn)
    # net_lite = CRNN_lite(imgH=32, nc=1, nclass=num_class, nh=256)
    # print(net_lite)
    crnn3 = CRNN_3(img_channel=3, img_height=32, img_width=128, num_class=num_class, map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False)
    print(crnn3)
    crnn3_cls = CRNN_3_cls(img_channel=3, img_height=64, img_width=160, num_class=num_class, map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False, num_regions=7)
    print(crnn3_cls)