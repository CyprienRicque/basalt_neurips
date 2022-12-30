from typing import Dict, Optional

import torch as th
from torch import nn
from torch.nn import functional as F

import src.lib.torch_util as tu
from src.lib.masked_attention import MaskedAttention
from src.lib.minecraft_util import store_args
from src.lib.tree_util import tree_map


def get_module_log_keys_recursive(m: nn.Module):
    """Recursively get all keys that a module and its children want to log."""
    keys = []
    if hasattr(m, "get_log_keys"):
        keys += m.get_log_keys()
    for c in m.children():
        keys += get_module_log_keys_recursive(c)
    return keys


class ImplementSubTasks(nn.Module):
    """Basic FCNN contatenatinng new inputs
    :param inchan: number of input channels
    :param outchan: number of output channels
    :param layer_args: positional layer args
    :param layer_type: options are "linear" (dense layer), "conv" (2D Convolution), "conv3d" (3D convolution)
    :param init_scale: multiplier on initial weights
    :param batch_norm: use batch norm after the layer (for 2D data)
    :param group_norm_groups: if not None, use group norm with this many groups after the layer. Group norm 1
        would be equivalent of layernorm for 2D data.
    :param layer_norm: use layernorm after the layer (for 1D data)
    :param layer_kwargs: keyword arguments for the layer
    """

    @store_args
    def __init__(
        self,
        inchan: int,
        outchan: int,
        hidsize: int,
        n_layers: int = 2,
        *layer_args,
        init_scale: int = 1,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation=True,
        log_scope: Optional[str] = None,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        layer = nn.Linear
        self.layers = nn.ModuleList(
            [
                layer(inchan if i == 0 else hidsize, outchan if i == (n_layers - 1) else hidsize,
                      bias=self.norm is None, *layer_args, **layer_kwargs)
                for i in range(n_layers)
            ]
        )

        # Init Weights (Fan-In)
        for layer in self.layers:
            layer.weight.data *= init_scale / layer.weight.norm(
                dim=tuple(range(1, layer.weight.data.ndim)), p=2, keepdim=True
            )
            # Init Bias
            if layer.bias is not None:
                layer.bias.data *= 0

    def forward(self, x):
        """Norm after the activation. Experimented with this for both IAM and BC and it was slightly better."""
        if self.norm is not None:
            x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
            if self.use_activation:
                x = F.relu(x, inplace=True)
            else:
                raise ValueError("FIXME Not using relu activation")
        return x

    def get_log_keys(self):
        return [
            f"activation_mean/{self.log_scope}",
            f"activation_std/{self.log_scope}",
        ]


class ImplementActions(nn.Module):
    """Basic FCNN contatenatinng new inputs
    :param inchan: number of input channels
    :param outchan: number of output channels
    :param layer_args: positional layer args
    :param layer_type: options are "linear" (dense layer), "conv" (2D Convolution), "conv3d" (3D convolution)
    :param init_scale: multiplier on initial weights
    :param batch_norm: use batch norm after the layer (for 2D data)
    :param group_norm_groups: if not None, use group norm with this many groups after the layer. Group norm 1
        would be equivalent of layernorm for 2D data.
    :param layer_norm: use layernorm after the layer (for 1D data)
    :param layer_kwargs: keyword arguments for the layer
    """

    @store_args
    def __init__(
        self,
        inchan: int,
        inchan_buttons: int,
        inchan_camera: int,
        outchan: int,
        hidsize_buttons: int,
        hidsize_camera: int,
        hidsize_concat: int,
        n_layers_actions: int = 2,
        n_layers_concat: int = 2,
        *layer_args,
        init_scale: int = 1,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation=True,
        log_scope: Optional[str] = None,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        layer = nn.Linear
        self.layers_buttons = nn.ModuleList(
            [
                layer(in_features=inchan_buttons if i == 0 else hidsize_buttons,
                      out_features=hidsize_buttons if i == (n_layers_actions - 1) else hidsize_buttons,
                      bias=self.norm is None, *layer_args, **layer_kwargs)
                for i in range(n_layers_actions)
            ]
        )
        self.layers_camera = nn.ModuleList(
            [
                layer(in_features=inchan_camera if i == 0 else hidsize_camera,
                      out_features=hidsize_camera if i == (n_layers_actions - 1) else hidsize_camera,
                      bias=self.norm is None, *layer_args, **layer_kwargs)
                for i in range(n_layers_actions)
            ]
        )
        self.layers_concat = nn.ModuleList(
            [
                layer(in_features=inchan + hidsize_buttons + hidsize_camera if i == 0 else hidsize_concat,
                      out_features=outchan if i == (n_layers_concat - 1) else hidsize_concat,
                      bias=self.norm is None, *layer_args, **layer_kwargs)
                for i in range(n_layers_concat)
            ]
        )

        # Init Weights (Fan-In)
        for layers in (self.layers_buttons, self.layers_camera, self.layers_concat):
            for layer in layers:
                layer.weight.data *= init_scale / layer.weight.norm(
                    dim=tuple(range(1, layer.weight.data.ndim)), p=2, keepdim=True
                )
                # Init Bias
                if layer.bias is not None:
                    layer.bias.data *= 0

    def forward(self, x, buttons, camera):
        for layer in self.layers_buttons:
            buttons = layer(buttons)
            if self.use_activation:
                buttons = F.relu(x, inplace=True)

        for layer in self.layers_camera:
            camera = layer(camera)
            if self.use_activation:
                camera = F.relu(x, inplace=True)

        if self.norm is not None:
            buttons = self.norm(buttons)
            camera = self.norm(camera)
            x = self.norm(x)

        for layer in self.layers_concat:
            x = layer(th.cat((x, buttons, camera), dim=2))
            if self.use_activation:
                x = F.relu(x, inplace=True)

        return x

    def get_log_keys(self):
        return [
            f"activation_mean/{self.log_scope}",
            f"activation_std/{self.log_scope}",
        ]


class FanInInitReLULayer(nn.Module):
    """Implements a slightly modified init that correctly produces std 1 outputs given ReLU activation
    :param inchan: number of input channels
    :param outchan: number of output channels
    :param layer_args: positional layer args
    :param layer_type: options are "linear" (dense layer), "conv" (2D Convolution), "conv3d" (3D convolution)
    :param init_scale: multiplier on initial weights
    :param batch_norm: use batch norm after the layer (for 2D data)
    :param group_norm_groups: if not None, use group norm with this many groups after the layer. Group norm 1
        would be equivalent of layernorm for 2D data.
    :param layer_norm: use layernorm after the layer (for 1D data)
    :param layer_kwargs: keyword arguments for the layer
    """

    @store_args
    def __init__(
        self,
        inchan: int,
        outchan: int,
        *layer_args,
        layer_type: str = "conv",
        init_scale: int = 1,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        use_activation=True,
        log_scope: Optional[str] = None,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)

        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, *layer_args, **layer_kwargs)

        # Init Weights (Fan-In)
        self.layer.weight.data *= init_scale / self.layer.weight.norm(
            dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
        )
        # Init Bias
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x):
        """Norm after the activation. Experimented with this for both IAM and BC and it was slightly better."""
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x

    def get_log_keys(self):
        return [
            f"activation_mean/{self.log_scope}",
            f"activation_std/{self.log_scope}",
        ]


class ResidualRecurrentBlocks(nn.Module):
    @store_args
    def __init__(
        self,
        n_block=2,
        recurrence_type="multi_layer_lstm",
        is_residual=True,
        **block_kwargs,
    ):
        super().__init__()
        init_scale = n_block ** -0.5 if is_residual else 1
        self.blocks = nn.ModuleList(
            [
                ResidualRecurrentBlock(
                    **block_kwargs,
                    recurrence_type=recurrence_type,
                    is_residual=is_residual,
                    init_scale=init_scale,
                    block_number=i,
                )
                for i in range(n_block)
            ]
        )

    def forward(self, x, first, ids):
        state_out = []
        # assert len(state) == len(
        #     self.blocks
        # ), f"Length of state {len(state)} did not match length of blocks {len(self.blocks)}"
        for block in self.blocks:
            x = block(x, first, ids)
        return x

    def initial_state(self, n_ids, batchsize) -> None:
        if "lstm" not in self.recurrence_type:
            for b in self.blocks:
                b.r.initial_state(n_ids, batchsize)

    def reset_states(self, ids: th.Tensor) -> None:
        if "lstm" not in self.recurrence_type:
            for b in self.blocks:
                b.r.reset_states(ids)


class ResidualRecurrentBlock(nn.Module):
    @store_args
    def __init__(
        self,
        hidsize,
        timesteps,
        init_scale=1,
        recurrence_type="multi_layer_lstm",
        is_residual=True,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        attention_heads=8,
        attention_memory_size=2048,
        attention_mask_style="clipped_causal",
        log_scope="resblock",
        block_number=0,
    ):
        super().__init__()
        self.log_scope = f"{log_scope}{block_number}"
        s = init_scale
        if use_pointwise_layer:
            if is_residual:
                s *= 2 ** -0.5  # second residual
            self.mlp0 = FanInInitReLULayer(
                hidsize,
                hidsize * pointwise_ratio,
                init_scale=1,
                layer_type="linear",
                layer_norm=True,
                log_scope=self.log_scope + "/ptwise_mlp0",
            )
            self.mlp1 = FanInInitReLULayer(
                hidsize * pointwise_ratio,
                hidsize,
                init_scale=s,
                layer_type="linear",
                use_activation=pointwise_use_activation,
                log_scope=self.log_scope + "/ptwise_mlp1",
            )

        self.pre_r_ln = nn.LayerNorm(hidsize)
        if recurrence_type in ["multi_layer_lstm", "multi_layer_bilstm"]:
            self.r = nn.LSTM(hidsize, hidsize, batch_first=True)
            raise NotImplementedError("state lstm pb")
            nn.init.normal_(self.r.weight_hh_l0, std=s * (self.r.weight_hh_l0.shape[0] ** -0.5))
            nn.init.normal_(self.r.weight_ih_l0, std=s * (self.r.weight_ih_l0.shape[0] ** -0.5))
            self.r.bias_hh_l0.data *= 0
            self.r.bias_ih_l0.data *= 0
        elif recurrence_type == "transformer":
            self.r = MaskedAttention(
                input_size=hidsize,
                timesteps=timesteps,
                memory_size=attention_memory_size,
                heads=attention_heads,
                init_scale=s,
                norm="none",
                log_scope=log_scope + "/sa",
                use_muP_factor=True,
                mask=attention_mask_style,
            )

    def forward(self, x, first, ids):
        residual = x
        x = self.pre_r_ln(x)
        x = recurrent_forward(
            self.r,
            x,
            first,
            ids,
            reverse_lstm=self.recurrence_type == "multi_layer_bilstm" and (self.block_number + 1) % 2 == 0,
        )

        if self.is_residual and "lstm" in self.recurrence_type:  # Transformer already residual.
            x = x + residual
        if self.use_pointwise_layer:
            # Residual MLP
            residual = x
            x = self.mlp1(self.mlp0(x))
            if self.is_residual:
                x = x + residual
        return x


def recurrent_forward(module, x, first, ids, reverse_lstm=False):
    if isinstance(module, nn.LSTM):
        raise NotImplementedError
        if state is not None:
            # In case recurrent models do not accept a "first" argument we zero out the hidden state here
            mask = 1 - first[:, 0, None, None].to(th.float)
            state = tree_map(lambda _s: _s * mask, state)
            state = tree_map(lambda _s: _s.transpose(0, 1), state)  # NL, B, H
        if reverse_lstm:
            x = th.flip(x, [1])
        x, state_out = module(x, state)
        if reverse_lstm:
            x = th.flip(x, [1])
        state_out = tree_map(lambda _s: _s.transpose(0, 1), state_out)  # B, NL, H
        return x, state_out
    else:
        return module(x, first, ids)


def _banded_repeat(x, t):
    """
    Repeats x with a shift.
    For example (ignoring the batch dimension):

    _banded_repeat([A B C D E], 4)
    =
    [D E 0 0 0]
    [C D E 0 0]
    [B C D E 0]
    [A B C D E]
    """
    b, T = x.shape
    x = th.cat([x, x.new_zeros(b, t - 1)], dim=1)
    result = x.unfold(1, T, 1).flip(1)
    return result


def bandify(b_nd, t, T):
    """
    b_nd -> D_ntT, where
        "n" indexes over basis functions
        "d" indexes over time differences
        "t" indexes over output time
        "T" indexes over input time
        only t >= T is nonzero
    B_ntT[n, t, T] = b_nd[n, t - T]
    """
    nbasis, bandsize = b_nd.shape
    b_nd = b_nd[:, th.arange(bandsize - 1, -1, -1)]
    if bandsize >= T:
        b_nT = b_nd[:, -T:]
    else:
        b_nT = th.cat([b_nd.new_zeros(nbasis, T - bandsize), b_nd], dim=1)
    D_tnT = _banded_repeat(b_nT, t)
    return D_tnT


def get_norm(name, d, dtype=th.float32):
    if name == "none":
        return lambda x: x
    elif name == "layer":
        return tu.LayerNorm(d, dtype=dtype)
    else:
        raise NotImplementedError(name)
