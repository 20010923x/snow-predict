import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, kernel_size=(3, 3), bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2    #paddding:用于保持输入和输出的空间维度不变，计算方式是卷积核大小的一半
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,   #输入通道数是Input_dim+hidden_dim
                              out_channels=4 * hidden_dim,   #输出通道数是4*hidden_dim（用于输入门、遗忘门、输出门和候选记忆单元）
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)   #一个二维卷积层

    def forward(self, input_tensor, cur_state):   #input_tensor：当前时间步的输入特征
        h_cur, c_cur = cur_state             #cur_state:包含前一时间步的隐藏状态h_cur和记忆单元c_cur
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 确保 input_tensor 和 h_cur 都在相同的设备上
        input_tensor = input_tensor.to(device)
        h_cur = h_cur.to(device)
        c_cur = c_cur.to(device)

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 假设 f, i, g, c_cur 已经定义
        f = f.to(device)
        i = i.to(device)
        g = g.to(device)
        c_cur = c_cur.to(device)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


conv_lstm_cell = ConvLSTMCell(input_dim=64, hidden_dim=256, kernel_size=(3, 3), bias=True)

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            self.cell_list.append(ConvLSTMCell(input_dim=input_dim,
                                               hidden_dim=hidden_dim[i],
                                               kernel_size=kernel_size[i],
                                               bias=bias))

    def forward(self, input_tensor, hidden_state=None):
        # 检查输入形状
        if input_tensor.dim() != 5:
            raise ValueError("Input tensor must be 5D, but got shape {}".format(input_tensor.shape))

        if not self.batch_first:
            # (seq_len, batch_size, channels, height, width) -> (batch_size, seq_len, channels, height, width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, seq_len, c, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w), device=input_tensor.device)

        else:
            hidden_state = tuple((h.to(input_tensor.device), c.to(input_tensor.device)) for h, c in hidden_state)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list
    def _init_hidden(self, batch_size, image_size, device):
        init_states = []
        for cell in self.cell_list:
            h = torch.zeros(batch_size, cell.hidden_dim, image_size[0], image_size[1], device=device)
            c = torch.zeros(batch_size, cell.hidden_dim, image_size[0], image_size[1], device=device)
            init_states.append((h, c))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
        #检查kernel_size是否为元组或元组列表

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    #如果不是列表，则将其扩展为列表，以匹配网络层数