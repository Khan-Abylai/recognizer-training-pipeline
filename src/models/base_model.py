import torch.nn as nn
import torch.nn.init as initializer
from torch.cuda.amp import autocast


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

    def __init__(self, type, num_class, lstm_layers, is_lstm_bidirectional, classification_regions=None):
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
        
        self.in_channel = 14 if type == 'extended' else 6
        self.squeeze = nn.Sequential(nn.Linear(512 * self.in_channel, 256), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.regions = classification_regions
        if classification_regions:
            num_regions = len(classification_regions)
            self.classificator = nn.Sequential(nn.Linear(9728, num_regions))

        initializer.xavier_uniform_(self.squeeze[0].weight)

        self.rnn = Embedding(256, 256, num_class, bidirectional=is_lstm_bidirectional, num_layers=lstm_layers)

    @autocast()
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 512 * self.in_channel, 38 if self.in_channel == 14 else 30).permute(0, 2, 1).contiguous()

        x = self.squeeze(x)

        if self.regions:
            cls = x.view(x.size(0), -1)
            return self.rnn(x), self.classificator(cls)
        return self.rnn(x), None
