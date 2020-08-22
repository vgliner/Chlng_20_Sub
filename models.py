import torch
import torch.nn as nn
import math


# An parent class for parallel modules
class ParallelModule(nn.Module):
    def __init__(self, modules: list, concatenation_dim=1):
        super().__init__()
        
        self.module_list = modules
        for i, module in enumerate(modules):
            name = 'fc' + str(i)
            self.add_module(name, module)
        self.concatenation_dim = concatenation_dim

    def forward(self, x: torch.Tensor, **kwargs):
        out = []
        for module in self.module_list:
            y = module(x)
            out.append(y)
        return torch.cat(out, dim=self.concatenation_dim)


# A simple but FF neural net
class SimpleFFN(nn.Module):
    def __init__(self, in_dim, num_of_classes, hidden_dims):
        super().__init__()
        
        layers = []
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)


# A simple but versatile d1 convolutional neural net
class ConvNet1d(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_lengths: list, dropout=None, stride=1, dilation=1, batch_norm=False,
                 flatten_output: bool = False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv1d(layer_in_channels, layer_out_channels, kernel_size=kernel_lengths[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)
        self.kernel_lengths = kernel_lengths
        self.stride = stride
        self.dilation = dilation
        self.hidden_channels = hidden_channels
        self.flatten = flatten_output

    def out_dim(self, in_length):
        out_channels = self.hidden_channels[-1]
        return out_channels * calc_out_length(in_length, self.kernel_lengths, self.stride, self.dilation)

    def forward(self, x):
        if not self.flatten:
            return self.cnn(x)
        else:
            return self.cnn(x).flatten(start_dim=1)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# A simple but versatile d2 convolutional neural net
class ConvNet2d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_sizes: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv2d(layer_in_channels, layer_out_channels, kernel_size=kernel_sizes[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm2d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# A simple but versatile d1 "deconvolution" neural net
class DeConvNet1d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, out_kernel: int,
                 kernel_lengths: list, dropout=None, stride=1, dilation=1, batch_norm=False, output_padding=1):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.ConvTranspose1d(layer_in_channels, layer_out_channels, kernel_size=kernel_lengths[i],
                                             stride=stride, dilation=dilation, output_padding=output_padding))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2))

            layer_in_channels = layer_out_channels

        layers.append(nn.ConvTranspose1d(layer_in_channels, out_channels, out_kernel, stride, dilation))

        self.dcnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dcnn(x)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Ecg12LeadNet(nn.Module):
    def forward(self, x):
        x1, x2 = x
        out1 = self.short_cnn(x1).reshape((x1.shape[0], -1))
        out2 = self.long_cnn(x2).reshape((x2.shape[0], -1))
        out = torch.cat((out1, out2), 1)
        return self.fc(out)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features        

    def __init__(self,
                 short_hidden_channels: list, long_hidden_channels: list,
                 short_kernel_lengths: list, long_kernel_lengths: list,
                 fc_hidden_dims: list,
                 short_dropout=None, long_dropout=None,
                 short_stride=1, long_stride=1,
                 short_dilation=1, long_dilation=1,
                 short_batch_norm=False, long_batch_norm=False,
                 short_input_length=1250, long_input_length=5000,
                 num_of_classes=2):
        super().__init__()
        assert len(short_hidden_channels) == len(short_kernel_lengths)
        assert len(long_hidden_channels) == len(long_kernel_lengths)

        self.short_cnn = ConvNet1d(12, short_hidden_channels, short_kernel_lengths, short_dropout,
                                   short_stride, short_dilation, short_batch_norm)
        self.long_cnn = ConvNet1d(1, long_hidden_channels, long_kernel_lengths, long_dropout,
                                  long_stride, long_dilation, long_batch_norm)

        short_out_channels = short_hidden_channels[-1]
        short_out_dim = short_out_channels * calc_out_length(short_input_length, short_kernel_lengths,
                                                             short_stride, short_dilation)
        long_out_channels = long_hidden_channels[-1]
        long_out_dim = long_out_channels * calc_out_length(long_input_length, long_kernel_lengths,
                                                           long_stride, long_dilation)

        in_dim = short_out_dim + long_out_dim
        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)


class Ecg12LeadMultiClassNet(nn.Module):
    def forward(self, x):
        x1, x2 = x
        out1 = self.short_cnn(x1).reshape((x1.shape[0], -1))
        out2 = self.long_cnn(x2).reshape((x2.shape[0], -1))
        out = torch.cat((out1, out2), 1)
        return self.fc(out)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features        

    def __init__(self,
                 short_hidden_channels: list, long_hidden_channels: list,
                 short_kernel_lengths: list, long_kernel_lengths: list,
                 fc_hidden_dims: list,
                 short_dropout=None, long_dropout=None,
                 short_stride=1, long_stride=1,
                 short_dilation=1, long_dilation=1,
                 short_batch_norm=False, long_batch_norm=False,
                 short_input_length=1250, long_input_length=5000,
                 num_of_classes=2):
        super().__init__()
        assert len(short_hidden_channels) == len(short_kernel_lengths)
        assert len(long_hidden_channels) == len(long_kernel_lengths)

        self.short_cnn = ConvNet1d(12, short_hidden_channels, short_kernel_lengths, short_dropout,
                                   short_stride, short_dilation, short_batch_norm)
        self.long_cnn = ConvNet1d(1, long_hidden_channels, long_kernel_lengths, long_dropout,
                                  long_stride, long_dilation, long_batch_norm)

        short_out_channels = short_hidden_channels[-1]
        short_out_dim = short_out_channels * calc_out_length(short_input_length, short_kernel_lengths,
                                                             short_stride, short_dilation)
        long_out_channels = long_hidden_channels[-1]
        long_out_dim = long_out_channels * calc_out_length(long_input_length, long_kernel_lengths,
                                                           long_stride, long_dilation)

        in_dim = short_out_dim + long_out_dim
        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)


# A naive multi-class CNN-FC model
class SimpleConvNetMulticlassModel(nn.Module):
    def forward(self, x):
        out = self.cnn(x).reshape((x.shape[0], -1))
        return self.fc(out)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __init__(self,
                 hidden_channels: list, kernel_lengths: list,
                 fc_hidden_dims: list, dropout=None,
                 stride=1, dilation=1, batch_norm=False,
                 input_length=1250, num_of_classes=2):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        self.cnn = ConvNet1d(12, hidden_channels, kernel_lengths, dropout, stride, dilation, batch_norm)

        out_channels = hidden_channels[-1]
        in_dim = out_channels * calc_out_length(input_length, kernel_lengths, stride, dilation)

        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)


# A model with common CNN for feature extraction and different FC net for each class
class FcForEachClass(nn.Module):
    def __init__(self,
                 hidden_channels: list, kernel_lengths: list,
                 fc_hidden_dims: list, dropout=None,
                 stride=1, dilation=1, batch_norm=False,
                 input_length=1250, num_of_classes=2):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        # Create the CNN
        self.cnn = ConvNet1d(12, hidden_channels, kernel_lengths, dropout, stride, dilation, batch_norm)

        out_channels = hidden_channels[-1]
        in_dim = out_channels * calc_out_length(input_length, kernel_lengths, stride, dilation)

        # Create the FC nets
        fc_nets = []
        for i in range(num_of_classes):
            fc_nets.append(SimpleFFN(in_dim, 1, fc_hidden_dims))

        self.fc = ParallelModule(fc_nets, 1)

    def forward(self, x):
        out = self.cnn(x).reshape((x.shape[0], -1))
        return self.fc(out)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def calc_out_length(l_in: int, kernel_lengths: list, stride: int, dilation: int, padding=0):
    l_out = l_in
    for kernel in kernel_lengths:
        l_out = math.floor((l_out + 2*padding - dilation * (kernel - 1) - 1) / stride + 1)
    return l_out


def calc_out_length_deconv(l_in: int, kernel_lengths: list, out_kernel: int, stride: int, dilation: int,
                           padding=0, output_padding=0):
    l_out = l_in
    for kernel in kernel_lengths:
        l_out = (l_out - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1
    l_out = (l_out - 1)*stride - 2*padding + dilation*(out_kernel - 1) + 1
    return l_out
