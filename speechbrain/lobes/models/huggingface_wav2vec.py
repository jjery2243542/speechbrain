"""This lobe enables the integration of huggingface pretrained wav2vec2 models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
"""

import torch
import torch.nn.functional as F
from torch import nn

# We check if transformers is installed.
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer

# For uniform initialization
BOUND = 0.01


class SimpleWav2Vec2Encoder(nn.Module):
    """
    copy from HuggingFace repo: https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
    """

    def __init__(self, config, n_layers):
        super().__init__()
        self.config = config
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                Wav2Vec2EncoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states, output_hidden_states=False):

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states, attention_mask=None, output_attentions=None
            )
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class HuggingFaceWav2Vec2MultiLayer(nn.Module):
    """This lobe enables the integration of HuggingFace
    pretrained wav2vec2.0 models.

    Source paper: https://arxiv.org/abs/2006.11477
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1,  768])
    """

    def __init__(
        self,
        source,
        save_path,
        add_n_layers=0,
        freeze=True,
        freeze_until_nth_layer=None,
        re_init_from_nth_layer=None,
        return_nth_layers=[-1],
        use_coef=False,
        temperature=1.0,
        apply_spec_augment=True,
        mask_time_prob=0.075,
        mask_time_length=10,
        mask_feature_prob=0.5,
        mask_feature_length=64,
        eval_mode=False,
    ):
        super().__init__()

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Download the model from HuggingFace.
        self.model = Wav2Vec2Model.from_pretrained(source, cache_dir=save_path)

        # Freeze feature extractor
        for p in self.model.feature_extractor.parameters():
            p.requires_grad = False

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        # Parameters for specaug
        self.model.config.apply_spec_augment = apply_spec_augment
        self.model.config.mask_time_prob = mask_time_prob
        self.model.config.mask_time_length = mask_time_length
        self.model.config.mask_feature_prob = mask_feature_prob
        self.model.config.mask_feature_length = mask_feature_length
        self.model.config.layerdrop = 0.0
        # self.model.config.attention_dropout = 0.0
        # self.model.config.feat_project_dropout = 0.0
        # self.model.config.hidden_dropout = 0.0

        # Make the model return multiple layers
        self.model.config.output_hidden_states = True
        self.hidden_size = self.model.config.hidden_size
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.num_hidden_states = self.model.config.num_hidden_layers + 1
        self.add_n_layers = add_n_layers

        # coefficient for each layers
        self.use_coef = use_coef
        self.temperature = temperature
        if self.use_coef:
            self.coef_param = nn.Parameter(torch.Tensor(self.num_hidden_states))
            nn.init.uniform_(self.coef_param, -BOUND, BOUND)
            self.softmax = nn.Softmax(dim=0)

        if self.use_coef and return_nth_layers[0] != -1:
            raise ValueError(
                "When using coefficient, the model will return the mixture of multiple layers."
            )

        self.return_nth_layers = [
            layer + self.num_hidden_states + self.add_n_layers
            if layer < 0
            else layer
            for layer in return_nth_layers
        ]
        if self.add_n_layers > 0:
            if not self.use_coef:
                extra_layers = [
                    Wav2Vec2EncoderLayer(self.model.config)
                    for _ in range(self.add_n_layers)
                ]
                self.model.encoder.layers.extend(extra_layers)
                if re_init_from_nth_layer is None:
                    re_init_from_nth_layer = self.model.config.num_hidden_layers
            else:
                self.extra_encoder = SimpleWav2Vec2Encoder(
                    config=self.model.config, n_layers=self.add_n_layers
                )

        # Randomly initialized layers if re_init_from_nth_layer is not None
        if re_init_from_nth_layer is not None:
            self.re_init_from_nth_layer = (
                re_init_from_nth_layer + self.num_hidden_layers
                if re_init_from_nth_layer < 0
                else re_init_from_nth_layer
            )

            self.reset_layers(self.model, self.re_init_from_nth_layer)

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        if freeze_until_nth_layer is not None:
            self.freeze_until_nth_layer = (
                freeze_until_nth_layer + self.num_hidden_layers
                if freeze_until_nth_layer < 0
                else freeze_until_nth_layer
            )
        if freeze:
            if freeze_until_nth_layer is None:
                self.freeze_layers(self.model, self.num_hidden_layers)
            else:
                self.freeze_layers(self.model, self.freeze_until_nth_layer)
        if eval_mode:
            self.model.eval()

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output for each layers
        out = self.model(wav)[1]
        if not self.use_coef:
            hiddens = [out[layer] for layer in self.return_nth_layers]
            hiddens = torch.cat(hiddens, dim=-1)
            return hiddens
        else:
            out = torch.stack(out, dim=2)
            coef = self.softmax(self.coef_param / self.temperature).view(
                1, 1, self.num_hidden_states, 1
            )
            out = torch.sum(coef * out, dim=2)
            if self.add_n_layers > 0:
                out = self.extra_encoder(out)
            return out

    def reset_layers(self, model, from_nth_layer):
        """Reinitializes part of the parameters of the network"""
        for layer in model.encoder.layers[from_nth_layer:]:
            layer.apply(model._init_weights)

    def freeze_layers(self, model, until_nth_layer):
        for layer in model.encoder.layers[:until_nth_layer]:
            for p in layer.parameters():
                p.requires_grad = False


class HuggingFaceWav2Vec2(nn.Module):
    """This lobe enables the integration of HuggingFace
    pretrained wav2vec2.0 models.

    Source paper: https://arxiv.org/abs/2006.11477
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1,  768])
    """

    def __init__(
        self, source, save_path, output_norm=True, freeze=True, pretrain=True
    ):
        super().__init__()

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Download the model from HuggingFace.
        # if pretrain is False, we do not download the pretrained weights
        # it it is True, we download and load them.
        if not (pretrain):
            config = Wav2Vec2Config.from_pretrained(source, cache_dir=save_path)
            self.model = Wav2Vec2Model(config)
        else:
            self.model = Wav2Vec2Model.from_pretrained(
                source, cache_dir=save_path
            )

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.output_norm = output_norm
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        out = self.model(wav)[0]

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out
