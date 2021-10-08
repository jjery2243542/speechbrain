import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.containers import Sequential, ConnectBlocks
from speechbrain.nnet.normalization import InstanceNorm1d
from speechbrain.nnet.pooling import AdaptivePool


class ConvBank(nn.Module):
    def __init__(self, input_shape, bank_size, channels):
        super(ConvBank, self).__init__()

        if bank_size % 2 == 0:
            raise ValueError("Bank size should be odd number.")

        self.convs = nn.ModuleList([])
        for kernel_size in range(1, bank_size, 2):
            self.convs.append(
                sb.nnet.CNN.Conv1d(
                    out_channels=channels,
                    kernel_size=kernel_size,
                    input_shape=input_shape,
                )
            )

    def forward(self, x):
        outputs = [x]
        for layer in self.convs:
            outputs.append(F.relu(layer(x)))

        outputs = torch.cat(outputs, dim=-1)
        return outputs


class AdativeInstanceNorm1d(nn.Module):
    def __init__(self, input_size, cond_size):
        super(AdativeInstanceNorm1d, self).__init__()
        self.norm = InstanceNorm1d(input_size=input_size)
        self.affine = sb.nnet.linear.Linear(
            n_neurons=input_size * 2, input_size=cond_size
        )

        # Set to dummy input
        self.cond = torch.zeros(1, 1, cond_size)

    # a hack to use sequential module
    def set_cond(self, cond):
        self.cond = cond

    def forward(self, x):
        out = self.norm(x)
        cond = self.affine(self.cond)
        sigma = cond[:, :, : cond.shape[-1] // 2]
        mu = cond[:, :, cond.shape[-1] // 2 :]
        out = out * sigma + mu
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_shape,
        kernel_size,
        channels,
        n_layers=2,
        cond_size=None,
        stride=1,
        upsample=1,
        norm_type=None,
        activation=nn.ReLU,
    ):
        super(ConvBlock, self).__init__()

        self.norm_type = norm_type
        self.cond_size = cond_size
        if self.cond_size is None and self.norm_type == "adain":
            raise ValueError("Must provide cond_size when using adain.")

        self.convs = Sequential(input_shape=input_shape)

        # If using adain, keep the instances of layers to set condition properly
        if self.norm_type == "adain":
            self.norm_layers = []

        self.stride = stride
        self.upsample = upsample

        for i in range(n_layers):
            self.convs.append(
                Conv1d,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride if i == n_layers - 1 else 1,
                layer_name=f"conv_{i}",
            )
            if self.norm_type == "in":
                self.convs.append(InstanceNorm1d(input_size=channels))
            elif self.norm_type == "adain":
                norm_layer = AdativeInstanceNorm1d(
                    input_size=channels, cond_size=cond_size
                )
                self.convs.append(norm_layer)
                self.norm_layers.append(norm_layer)
            self.convs.append(activation())

        # Add projection layer if dimension mixmatch
        self.project = False
        if channels != input_shape[-1]:
            self.project = True
            self.projection = sb.nnet.linear.Linear(
                n_neurons=channels, input_shape=input_shape
            )

        # For residual connection
        if self.stride > 1:
            self.pooling = sb.nnet.pooling.Pooling1d(
                "avg", kernel_size=stride, ceil_mode=True
            )

        # upsampling by linear layer
        if self.upsample > 1:
            self.upsampling = sb.nnet.linear.Linear(
                n_neurons=channels * self.upsample, input_size=channels
            )

    def forward(self, x, c=None):
        if self.norm_type == "adain" and c is None:
            raise ValueError("c has to be provided when using adain.")

        if self.norm_type == "adain":
            for layer in self.norm_layers:
                layer.set_cond(c)
        out = self.convs(x)

        if self.project:
            x = self.projection(x)
        if self.stride > 1:
            x = self.pooling(x)

        if self.upsample == 1:
            out = out + x

        if self.upsample > 1:
            upsampled = self.upsampling(out)
            out = upsampled.view(
                out.shape[0], out.shape[1] * self.upsample, out.shape[2]
            )
            out = out + F.interpolate(
                x.transpose(1, 2), scale_factor=self.upsample
            ).transpose(1, 2)

        return out


class DenseBlock(ConnectBlocks):
    def __init__(
        self, input_shape, n_layers=2, n_neurons=128, activation=nn.ReLU
    ):
        super(DenseBlock, self).__init__(
            input_shape=input_shape, shortcut_projection=True
        )
        for i in range(n_layers):
            self.append(sb.nnet.linear.Linear, n_neurons=n_neurons)
            if i == n_layers - 1:
                self.append(nn.ReLU(), end_of_block=True)
            else:
                self.append(nn.ReLU())


class SpeakerEncoder(Sequential):
    def __init__(
        self,
        input_shape,
        bank_size=7,
        n_conv_blocks=6,
        strides=[1, 2, 1, 2, 1, 2],
        n_dense_blocks=2,
        kernel_size=3,
        channels=128,
    ):
        super(SpeakerEncoder, self).__init__(input_shape=input_shape)
        self.append(ConvBank, bank_size=bank_size, channels=channels)

        for i in range(n_conv_blocks):
            self.append(
                ConvBlock,
                kernel_size=kernel_size,
                channels=channels,
                stride=strides[i],
            )
        self.append(AdaptivePool(output_size=1))

        for i in range(n_dense_blocks):
            self.append(DenseBlock, n_neurons=channels)


class ContentEncoder(Sequential):
    def __init__(
        self,
        input_shape,
        bank_size=7,
        n_conv_blocks=6,
        strides=[1, 2, 1, 2, 1, 2],
        n_dense_blocks=2,
        kernel_size=3,
        channels=128,
    ):
        super(ContentEncoder, self).__init__(input_shape=input_shape)
        self.append(ConvBank, bank_size=bank_size, channels=channels)

        for i in range(n_conv_blocks):
            self.append(
                ConvBlock,
                kernel_size=kernel_size,
                channels=channels,
                stride=strides[i],
                norm_type="in",
            )

        for i in range(n_dense_blocks):
            self.append(DenseBlock, n_neurons=channels)


class Decoder(nn.Module):
    def __init__(
        self,
        input_shape,
        cond_size=128,
        n_conv_blocks=6,
        upsamples=[1, 2, 1, 2, 1, 2],
        n_dense_blocks=2,
        kernel_size=3,
        channels=128,
    ):
        super(Decoder, self).__init__()
        self.conv_layers = nn.ModuleList([])

        shape = input_shape
        for i in range(n_conv_blocks):
            self.conv_layers.append(
                ConvBlock(
                    input_shape=shape,
                    cond_size=cond_size,
                    kernel_size=kernel_size,
                    channels=channels,
                    upsample=upsamples[i],
                    norm_type="adain",
                )
            )
            shape = (shape[0], shape[1] * upsamples[i], channels)

        self.dense_layers = Sequential(input_shape=shape)
        for i in range(n_dense_blocks):
            self.dense_layers.append(DenseBlock, n_neurons=channels)

    def forward(self, x, c):
        out = x
        for layer in self.conv_layers:
            out = layer(out, c=c)
        out = self.dense_layers(out)

        return out


class VCNet(nn.Module):
    def __init__(
        self,
        input_shape,
        bank_size=7,
        n_conv_blocks=6,
        kernel_size=3,
        channels=128,
        strides=[1, 2, 1, 2, 1, 2],
        upsamples=[1, 2, 1, 2, 1, 2],
        n_dense_blocks=2,
    ):
        super(VCNet, self).__init__()
        self.speaker_encoder = SpeakerEncoder(
            input_shape=input_shape,
            bank_size=bank_size,
            n_conv_blocks=n_conv_blocks,
            strides=strides,
            n_dense_blocks=n_dense_blocks,
            kernel_size=kernel_size,
            channels=channels,
        )
        self.content_encoder = ContentEncoder(
            input_shape=input_shape,
            bank_size=bank_size,
            n_conv_blocks=n_conv_blocks,
            strides=strides,
            n_dense_blocks=n_dense_blocks,
            kernel_size=kernel_size,
            channels=channels,
        )
        self.mu_layer = sb.nnet.linear.Linear(
            n_neurons=channels, input_size=channels
        )
        self.sigma_layer = sb.nnet.linear.Linear(
            n_neurons=channels, input_size=channels
        )
        self.decoder = Decoder(
            input_shape=self.content_encoder.get_output_shape(),
            n_conv_blocks=n_conv_blocks,
            upsamples=upsamples,
            n_dense_blocks=n_dense_blocks,
            kernel_size=kernel_size,
            channels=channels,
        )
        self.output_layer = sb.nnet.linear.Linear(
            n_neurons=input_shape[2], input_size=channels
        )

    def forward(self, x, y, noise=True):
        spk = self.speaker_encoder(y)
        con = self.content_encoder(x)
        mu = self.mu_layer(con)
        sigma = self.sigma_layer(con)
        if noise:
            eps = sigma.new(sigma.shape).normal_(0, 1)
            dec = self.decoder(mu + torch.exp(sigma / 2) * eps, spk)
        else:
            dec = self.decoder(mu, spk)
        out = self.output_layer(dec)

        # Remove additional frames
        out = out[:, : x.shape[1], :]
        return out, mu, sigma, con, spk
