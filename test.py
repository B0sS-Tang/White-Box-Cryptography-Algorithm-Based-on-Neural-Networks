#导入相关函数库
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,TensorDataset
import numpy as np


# 导入训练好的模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=2)  # 卷积
        self.pool1 = nn.MaxPool2d(2)  # 最大池化
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2)  # 卷积
        self.pool2 = nn.MaxPool2d(2)  # 最大池化

        self.linear1 = nn.Linear(5 * 13 * 32, 2024, bias=True)  # 两层全连接
        self.linear2 = nn.Linear(2024, 256, bias=False)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))  # 前向传播 包括激活函数
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 5 * 13 * 32)
        x = torch.tanh(self.linear1(x))
        output = F.softmax(self.linear2(x), dim=1)
        return output


net1 = torch.load('net1v2.1.pkl')
net2 = torch.load('net2v2.1.pkl')


# 导入训练好的模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=2)  # 卷积
        self.pool1 = nn.MaxPool2d(2)  # 最大池化
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2)  # 卷积
        self.pool2 = nn.MaxPool2d(2)  # 最大池化

        self.linear1 = nn.Linear(5 * 9 * 32, 2024, bias=True)  # 两层全连接
        self.linear2 = nn.Linear(2024, 256, bias=False)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))  # 前向传播 包括激活函数
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 5 * 9 * 32)
        x = torch.tanh(self.linear1(x))
        output = F.softmax(self.linear2(x), dim=1)
        return output


net3 = torch.load('net3v2.1.pkl')
xor = torch.load('xornet.pkl')

#输入待加密明文
print("请输入待加密明文：") #明文0123456789abcdeffedcba9876543210  默认密钥0f1571c947d9e8590cb7add6af7f6798
a=input('')
lenth=int(len(a)/2)
if(lenth!=16):
   print('输入长度不为16字节，请重新输入')
temp=[0 for i in range(lenth)]  #临时保存整形明文数组
for i in range(lenth):
    temp[i]=int(a[i*2:i*2+2],16)
#len(mingw) 1

# 加密主体，10轮加密
n = 0
while n < 10:
    # exor+byte+sbyte+sbyte2
    mingw = torch.zeros([lenth, 16 * 48], dtype=torch.float32)
    for i in range(lenth):
        mingw[i, 256 + 16 * n + i] = 1
        mingw[i, 512 + temp[i]] = 1
    plain = mingw.reshape([lenth, 1, 16, 48])
    plain = plain.cuda()
    etemp = net1(plain)
    # for i in range(lenth): #正确值 0x55 0x3b 0x81 0xd1 0x42 0x84 0xd9 0x8 0xff 0x26 0xc7 0xfb 0x2f 0xa4 0x19 0xd0 一致
    #    print(hex(torch.max(etemp,dim=1)[1][i]),end=' ')
    temp2 = torch.max(etemp, dim=1)[1]

    # 行移位
    temph = [0 for i in range(16)]
    for i in range(4):
        temph[i] = temp2[5 * i]
        temph[4 + i] = temp2[(4 + 5 * i) % 16]
        temph[8 + i] = temp2[(8 + 5 * i) % 16]
        temph[12 + i] = temp2[(12 + 5 * i) % 16]
        # for i in range(lenth): #0x55 0x84 0xc7 0xd0 0x42 0x26 0x19 0xd1 0xff 0xa4 0x81 0x8 0x2f 0x3b 0xd9 0xfb 一致
    #    print(hex(temph[i]),end=' ')
    # print('')

    # 列混淆
    if n != 9:
        index1 = [0 for i in range(16)]  # 分解运算
        index2 = [0 for i in range(16)]
        index3 = [0 for i in range(16)]
        index4 = [0 for i in range(16)]
        index1[0:16] = temph[0:16]
        for i in range(4):  # 将同目的序列聚在一起
            index2[4 * i] = temph[4 * i + 1]
            index2[4 * i + 1] = temph[4 * i + 2]
            index2[4 * i + 2] = temph[4 * i + 3]
            index2[4 * i + 3] = temph[4 * i]
            index3[4 * i] = temph[4 * i + 2]
            index3[4 * i + 1] = temph[4 * i + 3]
            index3[4 * i + 2] = temph[4 * i]
            index3[4 * i + 3] = temph[4 * i + 1]
            index4[4 * i] = temph[4 * i + 3]
            index4[4 * i + 1] = temph[4 * i]
            index4[4 * i + 2] = temph[4 * i + 1]
            index4[4 * i + 3] = temph[4 * i + 2]
        plain31 = torch.zeros([lenth * 4, 16 * 48], dtype=torch.float32)
        # ivsbyte2+ivsbyte+mul+skey
        for i in range(lenth):
            plain31[i, 0] = 1
            plain31[i, 256 + 16 * n + i] = 1
            plain31[i, 512 + index1[i]] = 1
            plain31[16 + i, 1] = 1
            plain31[16 + i, 256 + 16 * n + i] = 1
            plain31[16 + i, 512 + index2[i]] = 1
            plain31[32 + i, 2] = 1
            plain31[32 + i, 256 + 16 * n + i] = 1
            plain31[32 + i, 512 + index3[i]] = 1
            plain31[48 + i, 3] = 1
            plain31[48 + i, 256 + 16 * n + i] = 1
            plain31[48 + i, 512 + index4[i]] = 1
        plain31 = plain31.reshape([lenth * 4, 1, 16, 48])
        plain31 = plain31.cuda()
        tempby2 = net2(plain31)
        temp31 = torch.max(tempby2[0:16], dim=1)[1]
        temp32 = torch.max(tempby2[16:32], dim=1)[1]
        index3 = torch.max(tempby2[32:48], dim=1)[1]
        index4 = torch.max(tempby2[48:64], dim=1)[1]
        # for i in range(lenth): #0x51 0x6b 0x2c 0x9b 0x84 0x3b 0xa3 0x22 0x68 0x3d 0x30 0x3 0xf9 0xac 0xf 0x70
        #    print(hex(temp31[i]),end=' ')
        # print('')
        # for i in range(lenth): #0xdc 0xe0 0x80 0xee 0x8 0xda 0x77 0x77 0x69 0xec 0xd2 0x1f 0x9c 0xe7 0x0 0x71
        #    print(hex(temp32[i]),end=' ')
        # print('')

        templ = torch.zeros([lenth * 2, 512], dtype=torch.float32)
        for i in range(4):  # 左半异或
            templ[4 * i, temp31[4 * i]] = 1
            templ[4 * i, 256 + temp32[4 * i]] = 1
            templ[4 * i + 1, index4[4 * i + 1]] = 1
            templ[4 * i + 1, 256 + temp31[4 * i + 1]] = 1
            templ[4 * i + 2, index3[4 * i + 2]] = 1
            templ[4 * i + 2, 256 + index4[4 * i + 2]] = 1
            templ[4 * i + 3, temp32[4 * i + 3]] = 1
            templ[4 * i + 3, 256 + index3[4 * i + 3]] = 1
        for i in range(4):  # 右半异或
            templ[16 + 4 * i, index3[4 * i]] = 1
            templ[16 + 4 * i, 256 + index4[4 * i]] = 1
            templ[16 + 4 * i + 1, temp32[4 * i + 1]] = 1
            templ[16 + 4 * i + 1, 256 + index3[4 * i + 1]] = 1
            templ[16 + 4 * i + 2, temp31[4 * i + 2]] = 1
            templ[16 + 4 * i + 2, 256 + temp32[4 * i + 2]] = 1
            templ[16 + 4 * i + 3, index4[4 * i + 3]] = 1
            templ[16 + 4 * i + 3, 256 + temp31[4 * i + 3]] = 1
        templ = templ.reshape([lenth * 2, 1, 16, 32])
        templ = templ.cuda()
        texorl = xor(templ)
        xorl = torch.max(texorl[0:16], dim=1)[1]
        xorr = torch.max(texorl[16:32], dim=1)[1]
        plain3 = torch.zeros([lenth, 512], dtype=torch.float32)
        for i in range(4):  # 总异或
            plain3[4 * i, xorl[4 * i]] = 1
            plain3[4 * i, 256 + xorr[4 * i]] = 1
            plain3[4 * i + 1, xorl[4 * i + 1]] = 1
            plain3[4 * i + 1, 256 + xorr[4 * i + 1]] = 1
            plain3[4 * i + 2, xorl[4 * i + 2]] = 1
            plain3[4 * i + 2, 256 + xorr[4 * i + 2]] = 1
            plain3[4 * i + 3, xorl[4 * i + 3]] = 1
            plain3[4 * i + 3, 256 + xorr[4 * i + 3]] = 1
        plain3 = plain3.reshape([lenth, 1, 16, 32])
        plain3 = plain3.cuda()
        xtemp = xor(plain3)
        temp = torch.max(xtemp, dim=1)[1]
        print('第%d轮：' % (n + 1), end=' ')
        for i in range(lenth):  # 0xb9 0xe4 0x47 0xc5 0x94 0x8e 0x20 0xd6 0x57 0x16 0x9a 0xf5 0x75 0x51 0x3f 0x3b 一致
            print(hex(temp[i]), end=' ')
        print('')
    if n == 9:
        # ivsbyte2+ivsbyte+最后的exor
        plain4 = torch.zeros([lenth, 512], dtype=torch.float32)
        for i in range(lenth):
            plain4[i, i] = 1
            plain4[i, 256 + temph[i]] = 1
        plain4 = plain4.reshape([lenth, 1, 16, 32])
        plain4 = plain4.cuda()
        etemp = net3(plain4)
        temp4 = torch.max(etemp, dim=1)[1]
        print('第10轮：', end=' ')
        for i in range(lenth):
            print(hex(temp4[i]), end=' ')
        print('')
    n = n + 1