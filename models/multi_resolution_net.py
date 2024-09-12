# Copyright 2024 Takuya Fujimura

import torch
import torch.nn.functional as F
from relpath import add_import_path
from torch import nn

add_import_path("./")
from stft import FFT, STFT


def calc_filter_size(input_size, c, p, k, s, d=1):
    assert len(input_size) == 3
    return [
        c,
        int((input_size[1] + (2 * p) - (d * (k - 1)) - 1) / s) + 1,
        int((input_size[2] + (2 * p) - (d * (k - 1)) - 1) / s) + 1,
    ]


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))


class SEBlock(nn.Module):
    def __init__(self, num_channels, ratio=16):
        super().__init__()
        self.num_channels = num_channels
        self.ratio = ratio
        self.layer1 = nn.Sequential(
            nn.Linear(self.num_channels, self.num_channels // self.ratio, bias=False),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.num_channels // self.ratio, self.num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = torch.mean(x, dim=list(range(2, len(x.shape))))  # GAP
        w = self.layer1(w)
        w = self.layer2(w)
        w = w.view([x.shape[0], x.shape[1]] + ([1] * (len(x.shape) - 2)))
        return w * x


class SE3DBlock(nn.Module):
    def __init__(self, feat_size, ratio=4):
        super().__init__()
        self.ratio = ratio
        self.layer_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_feat, num_feat // self.ratio, bias=False),
                    nn.ReLU(),
                    nn.Linear(num_feat // self.ratio, num_feat, bias=False),
                    nn.Sigmoid(),
                )
                for num_feat in feat_size
            ]
        )

    def squeeze_excitation(self, x, dim):
        squeeze_dims = [i for i in range(1, 4) if i != dim]
        a = torch.mean(x, dim=squeeze_dims)
        a = self.layer_list[dim - 1](a)
        w = torch.sigmoid(a)
        unsqueezed_shape = [x.shape[0]]
        unsqueezed_shape += [1 if i != dim else x.shape[i] for i in range(1, 4)]
        w = w.reshape(unsqueezed_shape)
        return w * x

    def forward(self, x):
        """
        Args
            x: (B, 1, H, W)
        """
        for i in range(1, 4):
            x = x + self.squeeze_excitation(x, i)
        return x


class Conv1dEncoderLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        length,
        use_bias=False,
        emb_base_size=128,
        conv_param_list=[{"k": 256, "s": 64}, {"k": 64, "s": 32}, {"k": 16, "s": 4}],
    ):
        super().__init__()
        assert len(conv_param_list) == 3

        print("===Conv1dEncoderLayer==========")
        for i, param_dict in enumerate(conv_param_list):
            length = self.calc_conv1dsize(
                length, param_dict["k"], param_dict["s"], param_dict.get("d", 1)
            )
            print(f"{i}th: [{length}]")
        print("===============================")

        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=emb_base_size,
                kernel_size=conv_param_list[0]["k"],
                stride=conv_param_list[0]["s"],
                dilation=conv_param_list[0].get("d", 1),
                bias=use_bias,
            ),
            nn.ReLU(),
            SEBlock(emb_base_size, ratio=16),
            nn.Conv1d(
                in_channels=emb_base_size,
                out_channels=emb_base_size,
                kernel_size=conv_param_list[1]["k"],
                stride=conv_param_list[1]["s"],
                dilation=conv_param_list[1].get("d", 1),
                bias=use_bias,
            ),
            nn.ReLU(),
            SEBlock(emb_base_size, ratio=16),
            nn.Conv1d(
                in_channels=emb_base_size,
                out_channels=emb_base_size,
                kernel_size=conv_param_list[2]["k"],
                stride=conv_param_list[2]["s"],
                dilation=conv_param_list[2].get("d", 1),
                bias=use_bias,
            ),
            nn.ReLU(),
            SEBlock(num_channels=emb_base_size, ratio=16),
            nn.Flatten(),
            nn.Linear(length * emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
            nn.BatchNorm1d(emb_base_size),
            nn.ReLU(),
            nn.Linear(emb_base_size, emb_base_size, bias=use_bias),
        )

    def calc_conv1dsize(self, length, kernel, stride, dilation=1):
        return int((length - dilation * (kernel - 1) - 1) / stride + 1)
        # return int((length - kernel) / stride + 1)

    def forward(self, x):
        """
        x: (B, C, L)
        """
        x = self.layer(x)
        return x


class FFTEncoderLayer(nn.Module):
    def __init__(self, sec, sr, use_bias=False, emb_base_size=128):
        super().__init__()
        fft_len = (sec * sr) // 2 + 1
        conv_param_list = [{"k": 256, "s": 64}, {"k": 64, "s": 32}, {"k": 16, "s": 4}]
        self.layer = Conv1dEncoderLayer(
            1, fft_len, use_bias, emb_base_size, conv_param_list
        )

    def forward(self, x_fft):
        """
        x_fft: (B, L)
        """
        x_fft = x_fft.unsqueeze(1)
        return self.layer(x_fft)


class TimeEncoderLayer(nn.Module):
    def __init__(self, sec, sr, use_bias=False, emb_base_size=128):
        super().__init__()
        time_len = sec * sr
        conv_param_list = [{"k": 512, "s": 128}, {"k": 64, "s": 32}, {"k": 16, "s": 4}]
        self.layer = Conv1dEncoderLayer(
            1, time_len, use_bias, emb_base_size, conv_param_list
        )

    def forward(self, x_time):
        """
        x_fft: (B, L)
        """
        x_time = x_time.unsqueeze(1)
        return self.layer(x_time)


class STFT1dEncoderLayer(nn.Module):
    def __init__(self, spectrogram_size, dim, use_bias=False, emb_base_size=128):
        """
        Args
            dim: dimension of batch to which conv1d applies
        """
        super().__init__()
        assert dim in [1, 2]
        self.dim = dim
        self.bn = nn.BatchNorm1d(num_features=spectrogram_size[0])
        conv_param_list = [{"k": 64, "s": 16}, {"k": 32, "s": 16}, {"k": 16, "s": 4}]
        self.layer = Conv1dEncoderLayer(
            in_channel=spectrogram_size[2 - self.dim],
            length=spectrogram_size[self.dim - 1],
            use_bias=use_bias,
            emb_base_size=emb_base_size,
            conv_param_list=conv_param_list,
        )

    def forward(self, x):
        """
        x: (B, F, T)
        """
        x = self.bn(x)
        if self.dim == 1:
            x = x.permute(0, 2, 1)  # B, T, F
        return self.layer(x)


class ResNetBlock(nn.Module):
    def __init__(
        self, input_size, out_channel, kernel, stride, use_bias=False, se_mode="normal"
    ):
        """
        Args
            input_size: tuple ([C, H, W]) or int (C)
        """
        super().__init__()
        self.out_channel = out_channel
        self.kernel = kernel
        self.stride = stride

        if self.kernel != 3:
            raise NotImplementedError()

        if isinstance(input_size, int):
            assert se_mode == "normal"
            input_size = [input_size, -1, -1]

        self.bn = nn.BatchNorm2d(input_size[0])

        if stride == 1:
            self.skip_connect = None
        elif stride == 2:
            self.skip_connect = nn.Sequential(
                nn.MaxPool2d(kernel_size=self.kernel, stride=self.stride, padding=1),
                nn.Conv2d(
                    in_channels=input_size[0],
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=use_bias,
                ),
            )
        else:
            raise NotImplementedError()

        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_size[0],
                out_channels=self.out_channel,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=1,
                bias=use_bias,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.out_channel,
                out_channels=self.out_channel,
                kernel_size=self.kernel,
                stride=1,
                padding=1,
                bias=use_bias,
            ),
        )
        if se_mode == "normal":
            self.se = SEBlock(out_channel, ratio=16)
        elif se_mode == "3d":
            self.se = SE3DBlock(self.calc_outsize(input_size), ratio=4)
        else:
            raise NotImplementedError()

    def calc_outsize(self, input_size):
        feat_size = calc_filter_size(
            input_size, self.out_channel, 1, self.kernel, self.stride
        )
        feat_size = calc_filter_size(feat_size, self.out_channel, 1, self.kernel, 1)
        return feat_size

    def forward(self, x):
        x = self.bn(x)
        xr = self.layer(x)
        xr = self.se(xr)
        if self.skip_connect is not None:
            x = self.skip_connect(x) + xr
        else:
            x = x + xr
        return x


class FirstConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, use_bias=False):
        super().__init__()
        self.out_channel = out_channel
        self.kernel = kernel
        self.stride = stride
        self.pool_kernel = 3
        self.pool_stride = 2

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=self.out_channel,
                kernel_size=self.kernel,
                stride=self.stride,
                bias=use_bias,
                padding=0,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride),
        )

    def calc_convsize(self, input_size):
        return calc_filter_size(
            input_size, self.out_channel, 0, self.kernel, self.stride
        )

    def calc_poolsize(self, input_size):
        return calc_filter_size(
            input_size, input_size[0], 0, self.pool_kernel, self.pool_stride
        )

    def calc_outsize(self, input_size):
        return self.calc_poolsize(self.calc_convsize(input_size))

    def forward(self, x):
        return self.layer(x)


class STFT2dEncoderLayer(nn.Module):
    def __init__(
        self, spectrogram_size, use_bias=False, emb_base_size=128, se_mode="normal"
    ):
        super().__init__()
        assert emb_base_size % 8 == 0
        self.bn_freq = nn.BatchNorm1d(num_features=spectrogram_size[0])
        self.layers = nn.ModuleList()
        # 1st layer ######################################
        first_block = FirstConvBlock(1, emb_base_size // 8, 7, 2, use_bias)
        self.layers.append(first_block)
        feat_size = first_block.calc_outsize(
            [1, spectrogram_size[0], spectrogram_size[1]]
        )
        print("===STFT2dEncoderLayer============")
        print(f"1st: {feat_size}")
        # 2nd layer ######################################
        res1 = ResNetBlock(feat_size, emb_base_size // 8, 3, 1, use_bias, se_mode)
        feat_size = res1.calc_outsize(feat_size)
        res2 = ResNetBlock(feat_size, emb_base_size // 8, 3, 1, use_bias, se_mode)
        feat_size = res2.calc_outsize(feat_size)
        self.layers.append(nn.Sequential(res1, res2))
        print(f"2nd: {feat_size}")
        # 3, 4, 5th layer #############################
        for i, c in enumerate(
            [
                emb_base_size // 4,
                emb_base_size // 2,
                emb_base_size,
            ]
        ):
            res1 = ResNetBlock(feat_size, c, 3, 2, use_bias, se_mode)
            feat_size = res1.calc_outsize(feat_size)
            res2 = ResNetBlock(feat_size, c, 3, 1, use_bias, se_mode)
            feat_size = res2.calc_outsize(feat_size)
            self.layers.append(nn.Sequential(res1, res2))
            print(f"{i+3}th: {feat_size}")
        print("===============================")
        #################################################
        self.bn = nn.BatchNorm1d(emb_base_size * feat_size[1])
        self.linear = nn.Linear(
            emb_base_size * feat_size[1], emb_base_size, bias=use_bias
        )

    def forward(self, x):
        """
        Args
            x: spectrogram (B, F, T)
        """
        x = self.bn_freq(x).unsqueeze(1)

        for l in self.layers:
            x = l(x)  # B, C, F, T
        x = torch.max(x, dim=-1).values
        x = self.bn(torch.flatten(x, start_dim=1))
        x = self.linear(x)
        return x


class MultiResolutionModel(nn.Module):
    def __init__(
        self,
        sec,
        sr,
        stft_cfg_list,
        use_bias=False,
        use_time=False,
        emb_base_size=128,
        se_mode="normal",
    ):
        super().__init__()
        # STFT #########################################################
        self.stft_layer_list = nn.ModuleList([])
        for stft_cfg in stft_cfg_list:
            stft = STFT(**stft_cfg)
            spectrogram_size = stft(torch.randn(sec * sr)).shape
            if min(spectrogram_size) >= 36:
                stft_encoder = STFT2dEncoderLayer(
                    spectrogram_size, use_bias, emb_base_size, se_mode
                )
            elif spectrogram_size[0] >= 36 and spectrogram_size[1] < 36:
                stft_encoder = STFT1dEncoderLayer(
                    spectrogram_size, 1, use_bias, emb_base_size
                )
            elif spectrogram_size[0] < 36 and spectrogram_size[1] >= 36:
                stft_encoder = STFT1dEncoderLayer(
                    spectrogram_size, 2, use_bias, emb_base_size
                )
            else:
                raise ValueError("input sequence is too short")
            self.stft_layer_list.append(nn.Sequential(stft, stft_encoder))
        # FFT ##########################################################
        self.fft_layer = nn.Sequential(
            FFT(), FFTEncoderLayer(sec, sr, use_bias, emb_base_size)
        )
        # Time #########################################################
        if use_time:
            self.time_layer = TimeEncoderLayer(sec, sr, use_bias, emb_base_size)
            self.embedding_split_num = len(stft_cfg_list) + 2
        else:
            self.time_layer = None
            self.embedding_split_num = len(stft_cfg_list) + 1
        self.embedding_size = emb_base_size * self.embedding_split_num
        self.embedding_split_size = emb_base_size

    def forward(self, x_time):
        """
        Args
            x_time: (B, L)
        """
        z_list = [self.fft_layer(x_time)]

        if self.time_layer is not None:
            z_list += [self.time_layer(x_time)]

        for stft_layer in self.stft_layer_list:
            z_list += [stft_layer(x_time)]
        return torch.cat(z_list, dim=-1), z_list


###############################################################################################


def _module_test(n_fft, device="cuda:0"):
    x = torch.randn(2, 16000 * 18).to(device)
    stft_cfg = {
        "sr": 16000,
        "n_fft": n_fft,
        "hop_length": n_fft // 2,
        "n_mels": None,
        "power": 1.0,
        "use_mel": False,
        "f_min": 200.0,
        "f_max": 8000.0,
    }
    print("----------------------------------------------")
    net = MultiResolutionModel(18, 16000, [stft_cfg], use_time=True).to(device)
    y, _ = net(x)
    loss = torch.mean(y**2)
    loss.backward()
    print(y.shape)


if __name__ == "__main__":
    _module_test(8192)
    # _module_test(8192 * 2)
