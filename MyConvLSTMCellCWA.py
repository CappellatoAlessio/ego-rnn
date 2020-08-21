import torch
import torch.nn as nn


# from torch.autograd import Variable
# import torch.nn.functional as F

class MyConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, variant, kernel_size=3, stride=1, padding=1):
        super(MyConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.variant = variant
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_i_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_f_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_c_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_c_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_o_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        torch.nn.init.xavier_normal_(self.conv_i_xx.weight)
        torch.nn.init.constant_(self.conv_i_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_i_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_f_xx.weight)
        torch.nn.init.constant_(self.conv_f_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_f_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_c_xx.weight)
        torch.nn.init.constant_(self.conv_c_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_c_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_o_xx.weight)
        torch.nn.init.constant_(self.conv_o_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_o_hh.weight)

        if self.variant == 'c':
            self.conv_i = nn.Conv2d(hidden_size, hidden_size, kernel_size=1,
                                    bias=False) 
            torch.nn.init.xavier_normal_(self.conv_i.weight)
        if self.variant == 'd':
            self.conv_o = nn.Conv2d(hidden_size, hidden_size, kernel_size=1,
                                    bias=False)  
            torch.nn.init.xavier_normal_(self.conv_o.weight)

    def forward(self, x, state):
        if state is None:
            state = (torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).cuda(),
                     torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).cuda())
        ht_1, ct_1 = state
        if self.variant == 'c':
            zt = torch.exp(self.conv_i(torch.tanh(self.conv_i_xx(x) + self.conv_i_hh(ht_1))))
            zt_bz, zt_hz, zt_h, zt_w = zt.size()
            it = zt / torch.max(zt.view(zt_bz, zt_hz, -1), 2, True)[0].unsqueeze(3)
        else:
            it = torch.sigmoid(self.conv_i_xx(x) + self.conv_i_hh(ht_1))  # variant c
        ft = torch.sigmoid(self.conv_f_xx(x) + self.conv_f_hh(ht_1))
        ct_tilde = torch.tanh(self.conv_c_xx(x) + self.conv_c_hh(ht_1))
        ct = (ct_tilde * it) + (ct_1 * ft)
        if self.variant == 'd':
            zt = torch.exp(self.conv_o(torch.tanh(self.conv_o_xx(x) + self.conv_o_hh(ht_1))))
            zt_bz, zt_hz, zt_h, zt_w = zt.size()
            ot = zt / torch.max(zt.view(zt_bz, zt_hz, -1), 2, True)[0].unsqueeze(3)
        else:
            ot = torch.sigmoid(self.conv_o_xx(x) + self.conv_o_hh(ht_1))  # variant d
        ht = ot * torch.tanh(ct)
        return ht, ct
