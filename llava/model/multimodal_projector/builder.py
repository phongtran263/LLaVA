import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class AdditiveCouplingBlock(nn.Module):
    """Two-step additive coupling transform over the feature dimension."""

    def __init__(self, channels):
        super().__init__()
        if channels < 2:
            raise ValueError(
                "Additive coupling requires hidden_size to be at least 2, "
                f"but received {channels}."
            )

        self.first_channels = channels // 2
        self.second_channels = channels - self.first_channels
        self.first_update = self._conditioner(
            self.second_channels,
            self.first_channels,
        )
        self.second_update = self._conditioner(
            self.first_channels,
            self.second_channels,
        )

    @staticmethod
    def _conditioner(input_channels, output_channels):
        conditioner = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.GELU(),
            nn.Linear(output_channels, output_channels),
        )
        nn.init.zeros_(conditioner[-1].weight)
        nn.init.zeros_(conditioner[-1].bias)
        return conditioner

    def forward(self, x):
        first, second = torch.split(
            x,
            (self.first_channels, self.second_channels),
            dim=-1,
        )
        first = first + self.first_update(second)
        second = second + self.second_update(first)
        return torch.cat((first, second), dim=-1)


class CouplingProjector(nn.Module):
    """Project vision features, then refine them with additive coupling blocks."""

    def __init__(self, input_channels, output_channels, depth):
        super().__init__()
        if depth < 1:
            raise ValueError(
                "Coupling projector depth must be at least 1, "
                f"but received {depth}."
            )
        self.stem = nn.Linear(input_channels, output_channels)
        self.blocks = nn.ModuleList(
            AdditiveCouplingBlock(output_channels) for _ in range(depth)
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    coupling_gelu_match = re.match(r'^coupling(\d+)x_gelu$', projector_type)
    if coupling_gelu_match:
        coupling_depth = int(coupling_gelu_match.group(1))
        return CouplingProjector(
            config.mm_hidden_size,
            config.hidden_size,
            coupling_depth,
        )

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
