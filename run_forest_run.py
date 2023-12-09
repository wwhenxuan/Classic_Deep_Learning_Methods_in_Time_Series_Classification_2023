import numpy as np
import pandas as pd

from UCRdataset import DataSet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    """构建自己的Dataset对象"""

    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        word = self.words[idx]
        return word, label


def get_UCR_dataloader(name, train_num, test_num, one=0, train_batch_size=1, test_batch_size=1,
                       seed=np.random.randint(0, 500)):
    road = './UCRArchive/' + name + '/' + name + '_'
    df_train = pd.read_table(road + 'TRAIN.tsv', header=None)
    df_test = pd.read_table(road + 'TEST.tsv', header=None)

    train, test = {'samples': [], 'labels': []}, {'samples': [], 'labels': []}

    def append(dic, data):
        sample = data[1:]
        label = int(data[0]) - one
        if label == -1:
            label = 0
        dic['samples'].append(sample)
        dic['labels'].append(label)

    print("正在构建训练集")
    for i in tqdm(range(train_num)):
        data = np.array(df_train.loc[i])
        append(train, data)

    # 构建训练集
    print("正在构建验证集")
    for i in tqdm(range(test_num)):
        data = np.array(df_test.loc[i])
        append(test, data)

    for d in [train, test]:
        for key, value in d.items():
            if key != 'labels':
                d[key] = np.array(value)[:, np.newaxis]

    return (DataLoader(MyDataset(train['samples'], train['labels']), shuffle=True, batch_size=train_batch_size),
            DataLoader(MyDataset(test['samples'], test['labels']), shuffle=True, batch_size=test_batch_size))


class YourError(Exception):
    """自定义异常的类"""

    def __init__(self, message="nmd网络名称填错了"):
        super().__init__(message)


class ResBlock(nn.Module):
    """ResNet的残差模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3,), stride=(1,), padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3,), stride=(1,), padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3,), stride=(1,), padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

        self.channel = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(1,), stride=(1,), padding=0)
        )

    def forward(self, x):
        res = self.channel(x)
        x = self.conv(x)
        x = x + res
        return x


class ResNet(nn.Module):
    """ResNet网络框架 简单的基线"""

    def __init__(self, n_classes):
        super().__init__()

        self.ResBlock1 = ResBlock(in_channels=1, out_channels=64)
        self.ResBlock2 = ResBlock(in_channels=64, out_channels=128)
        self.ResBlock3 = ResBlock(in_channels=128, out_channels=128)

        self.GlobalPooling = nn.AdaptiveMaxPool1d(2)

        self.Fnn = nn.Linear(128 * 2, n_classes)

    def forward(self, x):
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)

        x = self.GlobalPooling(x)
        x = x.view(-1, 128 * 2)
        return self.Fnn(x)


class MultiBlock(nn.Module):
    """多尺度分支模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=(3,), stride=1, padding=1),
                                  nn.BatchNorm1d(num_features=out_channels),
                                  nn.ReLU(),
                                  nn.MaxPool1d(2))

    def forward(self, x):
        return self.conv(x)


class MCNN(nn.Module):
    """传统的多尺度卷积神经网络"""

    def __init__(self, n_classes, length, dropout=0.1):
        super().__init__()
        self.multi1 = MultiBlock(in_channels=1, out_channels=32)
        self.multi2 = MultiBlock(in_channels=1, out_channels=32)
        self.multi3 = MultiBlock(in_channels=1, out_channels=32)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=(3,), stride=(1,), padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            MultiBlock(in_channels=64, out_channels=128),
            MultiBlock(in_channels=128, out_channels=256))

        self.size = (length // 2 + length // 2 + length // 2 // 2) // 2 // 2

        self.fnn = nn.Sequential(
            nn.Linear(in_features=256 * self.size, out_features=256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_classes))

    def forward(self, data):
        x = data
        x = self.multi1(x)
        y = torch.flip(data, dims=[-1])
        y = self.multi2(y)
        z = torch.max_pool1d(data, kernel_size=2)
        z = self.multi3(z)
        inputs = torch.concat([x, y, z], dim=2)
        inputs = self.cnn(inputs)

        # print(inputs.size())
        # print(self.size)
        outputs = inputs.view(-1, 256 * self.size)
        return self.fnn(outputs)


class LSTM_FCN(nn.Module):
    """LSTM-FCN网络"""

    def __init__(self, n_classes, length):
        super().__init__()

        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128,
                      kernel_size=(3,), stride=(1,), padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=(3,), stride=(1,), padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128,
                      kernel_size=(3,), stride=(1,), padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=1, hidden_size=4, num_layers=12)
        self.down = nn.MaxPool1d(2)

        self.fnn = nn.Sequential(
            nn.Linear(4 * (length // 2) + 128 * 2, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        batch_size, _, length = x.size()
        lstm = x.permute(2, 0, 1)
        lstm, _ = self.lstm(lstm)
        lstm = lstm.permute(1, 2, 0)
        lstm = self.down(lstm)
        cnn = self.Conv(x)
        cnn = F.adaptive_max_pool1d(cnn, 2)

        lstm = lstm.reshape(batch_size, 4 * (length // 2))
        cnn = cnn.view(-1, 128 * 2)

        out = torch.concat([lstm, cnn], dim=1)
        return self.fnn(out)


class SEAttention1d(nn.Module):

    def __init__(self, channel, reduction):
        super(SEAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)

        return x * y.expand_as(x)


class macnn_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=None, stride=1, reduction=16):
        super(macnn_block, self).__init__()

        if kernel_size is None:
            kernel_size = [3, 6, 12]

        self.reduction = reduction

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size[0], stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size[1], stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size[2], stride=1, padding='same')

        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()

        self.se = SEAttention1d(out_channels * 3, reduction=reduction)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x_con = torch.cat([x1, x2, x3], dim=1)

        out = self.bn(x_con)
        out = self.relu(out)

        out_se = self.se(out)

        return out_se


class MACNN(nn.Module):
    """注意力多尺度卷积神经网络"""

    def __init__(self, n_classes, in_channels=1, channels=64, block_num=None):
        super(MACNN, self).__init__()

        if block_num is None:
            block_num = [2, 2, 2]

        self.in_channel = in_channels
        self.num_classes = n_classes
        self.channel = channels

        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.channel * 12, n_classes)

        self.layer1 = self._make_layer(macnn_block, block_num[0], self.channel)
        self.layer2 = self._make_layer(macnn_block, block_num[1], self.channel * 2)
        self.layer3 = self._make_layer(macnn_block, block_num[2], self.channel * 4)

    def _make_layer(self, block, block_num, channel, reduction=16):

        layers = []
        for i in range(block_num):
            layers.append(block(self.in_channel, channel, kernel_size=None,
                                stride=1, reduction=reduction))
            self.in_channel = 3 * channel

        return nn.Sequential(*layers)

    def forward(self, x):

        out1 = self.layer1(x)
        out1 = self.max_pool1(out1)

        out2 = self.layer2(out1)
        out2 = self.max_pool2(out2)

        out3 = self.layer3(out2)
        out3 = self.avg_pool(out3)

        out = torch.flatten(out3, 1)
        out = self.fc(out)

        return out


class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)


def pass_through(X):
    return X


class Inception(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(),
                 return_indices=False):
        super(Inception, self).__init__()
        self.return_indices = return_indices
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, X):
        Z_bottleneck = self.bottleneck(X)
        if self.return_indices:
            Z_maxpool, indices = self.max_pool(X)
        else:
            Z_maxpool = self.max_pool(X)
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        Z = self.activation(self.batch_norm(Z))
        if self.return_indices:
            return Z, indices
        else:
            return Z


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True,
                 activation=nn.ReLU(), return_indices=False):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.inception_1 = Inception(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_2 = Inception(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_3 = Inception(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                )
            )

    def forward(self, X):
        if self.return_indices:
            Z, i1 = self.inception_1(X)
            Z, i2 = self.inception_2(Z)
            Z, i3 = self.inception_3(Z)
        else:
            Z = self.inception_1(X)
            Z = self.inception_2(Z)
            Z = self.inception_3(Z)
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z


class InceptionTime(nn.Module):
    """Inception调试网络"""

    def __init__(self, n_classes):
        super().__init__()

        self.block = nn.Sequential(
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
        )

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = Flatten(out_features=32 * 4 * 1)
        self.fnn = nn.Sequential(
            nn.Linear(in_features=32 * 4 * 1,
                      out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,
                      out_features=n_classes)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fnn(x)


class MLP(nn.Module):
    """TSC的多层感知机"""

    def __init__(self, n_classes, length):
        super().__init__()

        self.fnn = nn.Sequential(
            nn.Linear(in_features=length, out_features=500),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=500),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=500),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=n_classes),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, length = x.size()
        x = x.view(-1, length)
        return self.fnn(x)


class FCN(nn.Module):
    """TSC的全连接神经网络"""

    def __init__(self, n_classes):
        super().__init__()

        self.fcn = self._get_conv()
        self.global_maxpool = nn.AdaptiveAvgPool1d(4)
        self.fnn = nn.Linear(in_features=128 * 4, out_features=n_classes)

    @staticmethod
    def _get_conv():
        module = []
        for channels in [(1, 128), (128, 256), (256, 128)]:
            module.append(nn.Conv1d(in_channels=channels[0], out_channels=channels[1],
                                    kernel_size=(3,), stride=(1,), padding=1))
            module.append(nn.BatchNorm1d(num_features=channels[1]))
            module.append(nn.ReLU(inplace=True))
        return nn.Sequential(*module)

    def forward(self, x):
        x = self.fcn(x)
        x = self.global_maxpool(x)
        # print(x.size())
        x = x.view(-1, 128 * 4)
        return self.fnn(x)


def train_network(net, num_epochs, optimizer, scheduler, criterion,
                  train_loader, test_loader,
                  batch_size, length, test_epochs):
    """训练神经网络所用的函数"""

    count = 0
    train_loss = np.zeros([num_epochs, 1])
    train_acc = np.zeros([num_epochs, 1])
    test_acc = np.zeros([num_epochs, 1])

    for epoch in range(num_epochs):
        net.train()
        loss_x = 0
        with tqdm(total=length, desc=f'Epoch {epoch + 1}') as pbar:
            for i, train_data in enumerate(train_loader, 1):
                data, labels = train_data
                data = data.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                predict_label = net(data)
                loss = criterion(predict_label, labels)
                loss_x += loss.item()
                loss.backward()
                optimizer.step()
                count += 1
                pbar.update(1)
            train_loss[epoch] = loss_x / (batch_size * i)

            net.eval()
            total_correct_train = 0
            total_samples_train = 0
            total_correct_test = 0
            total_samples_test = 0

            with torch.no_grad():
                for batch_data in train_loader:
                    inputs, label = batch_data
                    inputs = inputs.cuda()
                    label = label.cuda()
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_correct_train += (predicted == label).sum().item()
                    total_samples_train += label.size(0)

                if epoch >= test_epochs:
                    for batch_data in test_loader:
                        # 取测试数据
                        inputs, label = batch_data
                        inputs = inputs.cuda()
                        label = label.cuda()
                        # 前向传播
                        outputs = net(inputs)
                        # 计算准确率
                        _, predicted = torch.max(outputs, 1)
                        total_correct_test += (predicted == label).sum().item()
                        total_samples_test += label.size(0)

            acc_train = total_correct_train / total_samples_train
            train_acc[epoch] = acc_train
            if epoch >= test_epochs:
                acc_test = total_correct_test / total_samples_test
                test_acc[epoch] = acc_test

            pbar.set_postfix({'Train count': count,
                              'Loss': train_loss[epoch][0],
                              'Train acc': str(train_acc[epoch][0] * 100)[: 6] + '%',
                              'Test acc': str(test_acc[epoch][0] * 100)[: 6] + '%'})

        scheduler.step()

    print("Highest Accuracy:", test_acc.max())
    return test_acc.max()


class Net(object):
    """训练网络网络通用的接口"""

    def __init__(self, network, name, lr, count,
                 train_batch_size, test_batch_size, epochs, test_epochs=10,
                 step_size=15, gamma=0.5):
        self.network = network
        self.name = name
        self.lr = lr
        self.count = count
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.test_epochs = test_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.dataset = DataSet[self.name]

        self.n_classes = self.dataset['n_classes']
        self.length = self.dataset['length']
        self.train_num = self.dataset['train_num']
        self.test_num = self.dataset['test_num']
        self.one = self.dataset['one']

        self.acc = np.zeros(self.count)
        self._train_network()

    def _get_dataloader(self):
        return get_UCR_dataloader(name=self.name,
                                  train_num=self.train_num,
                                  test_num=self.test_num,
                                  train_batch_size=self.train_batch_size,
                                  test_batch_size=self.test_batch_size,
                                  one=self.one)

    def _get_network(self):
        print("正在获取网络:" + self.network)
        if self.network == 'MLP':
            return MLP(n_classes=self.n_classes, length=self.length)
        elif self.network == 'FCN':
            return FCN(n_classes=self.n_classes)
        elif self.network == 'ResNet':
            return ResNet(n_classes=self.n_classes)
        elif self.network == 'LSTM-FCN':
            return LSTM_FCN(n_classes=self.n_classes, length=self.length)
        elif self.network == 'MCNN':
            return MCNN(n_classes=self.n_classes, length=self.length)
        elif self.network == 'MACNN':
            return MACNN(n_classes=self.n_classes)
        elif self.network == 'InceptionTime':
            return InceptionTime(n_classes=self.n_classes)
        else:
            raise YourError

    def _train_network(self):
        for i in range(1, self.count + 1):
            print(f"第{i}轮网络训练")
            train_loader, test_loader = self._get_dataloader()
            net = self._get_network().double().cuda()
            optimizer = torch.optim.Adam(net.parameters())
            # scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            criterion = nn.CrossEntropyLoss(size_average=False)
            acc = train_network(net=net, num_epochs=self.epochs, optimizer=optimizer, criterion=criterion,
                                train_loader=train_loader, test_loader=test_loader, test_epochs=self.test_epochs,
                                length=int(self.train_num / self.train_batch_size),
                                batch_size=self.train_batch_size, scheduler=scheduler)
            self.acc[i - 1] = acc

    def print_acc(self):
        print(self.acc)
        print("均值", self.acc.mean(), "标准差", self.acc.std())


print("Hello, world!")
