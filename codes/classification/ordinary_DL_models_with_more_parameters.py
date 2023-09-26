import torch
from torch import nn
# import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class WordAveraging2(nn.Module):
    def __init__(self, vocab_size, embed_size, num_of_classes, padding_index, pretrained_weight=False):
        super(WordAveraging2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)
        if not isinstance(pretrained_weight, bool):
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.linear = nn.Linear(embed_size, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, num_of_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        # x = x.detach()
        x = torch.mean(x, dim=1)
        outputs = self.linear(x)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        return self.softmax(outputs)


class TextCNN2(nn.Module):
    def __init__(self, vocab_size, embed_size, num_of_filters, window_sizes, max_len, num_of_classes, padding_index, pretrained_weight=False):
        super(TextCNN2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)
        if not isinstance(pretrained_weight, bool):
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(embed_size, num_of_filters, h),
                           nn.ReLU(),
                           nn.MaxPool1d(max_len - h + 1)
            ) for h in window_sizes]
        )
        self.linear = nn.Linear(num_of_filters * len(window_sizes), 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, num_of_classes)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        x = self.embedding(x)
        # x = x.detach()
        x = x.permute(0, 2, 1)
        # x = torch.Tensor(x).cuda()
        x = Variable(x)
        # print(torch)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(-1, x.size(1))
        outputs = self.linear(x)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        return self.softmax(outputs)

class TextLSTM2(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_of_classes, padding_index, pretrained_weight=False):
        super(TextLSTM2, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)
        if not isinstance(pretrained_weight, bool):
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)

        self.f1 = nn.Sequential(nn.Linear(hidden_size * 2, 200),
                                nn.Dropout(0.8),
                                nn.ReLU(),
                                nn.Linear(200, num_of_classes),
                                nn.Softmax(dim=-1)
                                )

    def forward(self, x):
        x = self.embedding(x)
        # x = x.detach().numpy()
        x = x.permute(1, 0, 2)
        h0 = torch.randn(2, x.size(1), self.hidden_size).cuda()
        c0 = torch.randn(2, x.size(1), self.hidden_size).cuda()
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # output, (hn, cn) = self.lstm(x)
        # print(hn.shape())
        # hn = torch.cat((hn[0], hn[1]), 1)
        # print(hn, output[-1])
        # print(hn.shape, output[-1].shape)
        # x = self.f1(hn)
        x = self.f1(output[-1])
        return x


class TextLSTMAttention2(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_of_classes, padding_index, pretrained_weight=False):
        super(TextLSTMAttention2, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)
        if not isinstance(pretrained_weight, bool):
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)

        # 定义计算注意力权重的内容
        self.linear1 = nn.Linear(hidden_size * 2, 200)
        self.tanh = nn.Tanh()
        self.u_w = nn.Linear(200, 1)
        self.softmax1 = nn.Softmax(dim=-1)

        # 定义输出
        self.f1 = nn.Sequential(nn.Linear(hidden_size * 2, 200),
                                nn.Dropout(0.8),
                                nn.ReLU(),
                                nn.Linear(200, num_of_classes),
                                nn.Softmax(dim=-1)
                                )

    def forward(self, x):
        x = self.embedding(x)
        # x = x.detach().numpy()
        x = x.permute(1, 0, 2)
        h0 = torch.randn(2, x.size(1), self.hidden_size).cuda()
        c0 = torch.randn(2, x.size(1), self.hidden_size).cuda()
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output.permute(1, 0, 2)
        # (seq_len, batchsize, hidden_size*num_directions) => (batchsize, seq_len, hidden_size*num_directions)
        attention_u = self.tanh(self.linear1(output))
        # (batchsize, seq_len, hidden_size*num_directions) => (batchsize, seq_len, u_size)
        attention_a = self.softmax1(self.u_w(attention_u))
        # print(attention_a.shape, output.shape)
        # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
        output = torch.matmul(attention_a.permute(0, 2, 1), output).squeeze()
        # (batchsize, 1, seq_len) * (batchsize, seq_len, hidden_size * num_directions) =>(batchsize, hidden_size * num_directions)
        return self.f1(output)


class TextLSTMCNN2(nn.Module):
    def __init__(self, vocab_size, embed_size, num_of_filters, window_sizes, max_len, hidden_size, num_of_classes, padding_index, pretrained_weight=False):
        super(TextLSTMCNN2, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)
        if not isinstance(pretrained_weight, bool):
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(embed_size + 2 * self.hidden_size, num_of_filters, h),
                           nn.ReLU(),
                           nn.MaxPool1d(max_len - h + 1)
                           ) for h in window_sizes]
        )
        # self.linear = nn.Linear(num_of_filters * len(window_sizes), num_of_classes)
        self.linear = nn.Linear(num_of_filters * len(window_sizes), 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, num_of_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        # x = x.detach().numpy()
        input = x.permute(1, 0, 2)
        h0 = torch.randn(2, input.size(1), self.hidden_size).cuda()
        c0 = torch.randn(2, input.size(1), self.hidden_size).cuda()
        output, (hn, cn) = self.lstm(input, (h0, c0))
        output = output.permute(1, 0, 2)
        # print(output.shape)
        outputsplit1, outputsplit2 = output.chunk(2, dim=2)
        # print(outputsplit2.shape)
        outputcat = torch.cat((outputsplit1, x, outputsplit2), dim=2)
        outputcat = outputcat.permute(0, 2, 1)
        x = [conv(outputcat) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.softmax(x)


