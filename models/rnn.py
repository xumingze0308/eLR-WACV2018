import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,
                 input_size=4096,
                 hidden_size=512,
                 bidirectional=True,
                 num_classes=51):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        self.rnn1 = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.rnn2 = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.classifier = \
                nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x1, x2):
        x1, hx1 = self.rnn1(x1, None)
        x2, hx2 = self.rnn2(x2, None)
        hx = hx1[0,:,:] + hx2[0,:,:]
        if self.bidirectional:
            hx = hx + hx1[1,:,:] + hx2[1,:,:]
        hx = self.classifier(hx)
        return hx

