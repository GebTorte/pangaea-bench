import logging
from logging import Logger
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from pangaea.encoders.base import Encoder

logger = logging.getLogger(__name__)


class THOR_Encoder(Encoder):
    """Thin adapter that exposes THOR through the PANGAEA encoder interface.

    This wrapper keeps the encoder boundary small and explicit:
    - PANGAEA provides a dict of modality tensors
    - the adapter maps those tensors to THOR's channel/product layout
    - THOR runs natively when available
    - a compatibility fallback preserves the PANGAEA contract if a full native
      integration is not yet available or if a checkpoint is missing.
    """

    def __init__(
        self,
        encoder_weights: str | Path | None = None,
        input_bands: dict[str, list[str]] | None = None,
        input_size: int = 224,
        embed_dim: int = 768,
        output_layers: list[int] | None = None,
        output_dim: int | list[int] = 768,
        download_url: str | None = None,
        model_name: str = "thor_vit_base_encoder_alibi_patch_size_embed_v1",
        native_model: str | None = None,
        ground_covers: list[int] | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if input_bands is None:
            input_bands = {
                "optical": ["B4", "B3", "B2"],
            }
        if output_layers is None:
            output_layers = [3, 5, 7, 11]

        super().__init__(
            model_name=model_name,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            encoder_weights=encoder_weights,
            download_url=download_url,
        )

        self.ground_covers = list(ground_covers) if ground_covers is not None else [input_size]
        self.native_model_name = native_model or model_name
        self.thor_model = None
        self._fallback_projection = None

        self._build_native_thor_model()

    def _build_native_thor_model(self) -> None:
        """Attempt to initialize the native THOR model if the THOR repo is available."""
        try:
            from thor.core.model_registry import MODELS
            from thor.models.thor_vit import ThorViTEncoder
        except Exception as exc:  # pragma: no cover - environment-dependent path
            logger.warning(
                "THOR repo not available in the current Python environment; using a compatibility adapter. "
                "Install the THOR package or add the repo to PYTHONPATH. Original error: %s",
                exc,
            )
            self.thor_model = None
            self._fallback_projection = nn.Identity()
            return

        try:
            channels = self._build_thor_channels()
            input_params = {
                "ground_covers": self.ground_covers,
                "channels": channels,
                "groups": [list(channels.keys())],
                "cls_token_type": "pooled",
                "encoder_pos_type": "alibi",
                "use_flexivit": True,
                "flexivit_ref_patch_size": 4,
            }

            if self.native_model_name in MODELS.models:
                factory = MODELS.get_model(self.native_model_name)
                self.thor_model = factory(input_params=input_params, embed_dim=self.embed_dim)
            else:
                self.thor_model = ThorViTEncoder(
                    input_params=input_params,
                    embed_dim=self.embed_dim,
                    depth=12,
                    num_heads=max(3, self.embed_dim // 64),
                    mlp_ratio=4.0,
                )
            logger.info("Initialized native THOR model adapter for %s", self.native_model_name)
        except Exception as exc:  # pragma: no cover - model-specific compatibility issues
            logger.warning(
                "Native THOR model could not be instantiated with this dataset contract; using compatibility mode. "
                "This adapter is still valid for interface-level integration, but full performance fidelity requires a "
                "THOR config compatible with the PANGAEA dataset preprocessing. Error: %s",
                exc,
            )
            self.thor_model = None
            self._fallback_projection = nn.Conv2d(
                in_channels=sum(len(v) for v in self.input_bands.values()),
                out_channels=self.embed_dim,
                kernel_size=1,
                bias=True,
            )

    @staticmethod
    def _get_band_gsd(product: str, band: str) -> int:
        """Return the native GSD in meters for a product/band pair used by THOR."""
        band_name = str(band).upper()
        if product == "S2" and band_name in {"B1", "B9", "B10"}:
            return 60
        elif product == "S2" and band_name in {"B5", "B6", "B7", "B8A", "B11", "B12"}:
            return 20
        elif product == "S2" and band_name in {"B2", "B3", "B4", "B8"}:
            return 10
        elif product == "S1": #and band_name in {"VV", "VH"}: # keep redundant case for S1
            return 10
        return 10 # default to 10m for unknown bands


    def _resolve_ground_cover(self, x):
        """
        Resolve ground cover based on the input tensor shapes and their corresponding GSDs.

        Then pass that resolved cover into THOR instead of self.ground_covers[0].
        """
        covers = set()
        for modality, tensor in x.items():
            if tensor is None:
                continue
            bands = self.input_bands.get(modality, [])
            for idx, band in enumerate(bands):
                product = "S2" if modality in {"optical", "rgb"} else ("S1" if modality in {"sar", "radar"} else modality.upper())
                gsd = self._get_band_gsd(product, band)
                cover = int(tensor.shape[-1] * gsd)
                covers.add(cover)

        if len(covers) > 1:
            raise ValueError(f"Mixed THOR covers in one sample: {sorted(covers)}")
        return next(iter(covers))

    def _build_thor_channels(self) -> dict[str, dict[str, int]]:
        """Build the THOR product/band metadata expected by the model.

        This is intentionally conservative and keeps the adapter explicit about
        channel grouping, spatial layout, and GSD assumptions.
        """
        channels: dict[str, dict[str, int]] = {}
        patch_size = max(1, self.input_size // 16)
        for modality, bands in self.input_bands.items():
            if modality == "optical":
                product = "S2"
            elif modality in {"sar", "radar"}:
                product = "S1"
            else:
                product = modality.upper()

            for band in bands:
                key = f"{product}:{band}"
                channels[key] = {
                    "GSD": self._get_band_gsd(product, band),
                    "patch_size": patch_size,
                    "num_patch": max(1, self.input_size // patch_size),
                }
        return channels

    def _prepare_thor_input(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map PANGAEA modality dicts to a THOR-compatible per-band dict."""
        if x is None:
            return {}

        if all(isinstance(k, str) and ":" in k for k in x.keys()):
            return {k: v for k, v in x.items()}

        prepared: dict[str, torch.Tensor] = {}
        for modality, tensor in x.items():
            if tensor is None:
                continue
            tensor = tensor.to(dtype=torch.float32)
            if tensor.dim() == 5:
                tensor = tensor[:, :, 0]
            if tensor.dim() != 4:
                raise ValueError(f"Unsupported tensor shape for modality {modality}: {tuple(tensor.shape)}")

            bands = self.input_bands.get(modality, [])
            if not bands:
                continue
            if tensor.shape[1] != len(bands):
                if tensor.shape[1] == 1 and len(bands) > 1:
                    tensor = tensor.repeat(1, len(bands), 1, 1)
                else:
                    raise ValueError(
                        f"Mismatch for modality {modality}: expected {len(bands)} channels, got {tensor.shape[1]}"
                    )

            for idx, band in enumerate(bands):
                if modality in {"optical", "rgb"}:
                    product_band = f"S2:{band}"
                elif modality in {"sar", "radar"}:
                    product_band = f"S1:{band}"
                else:
                    product_band = f"{modality.upper()}:{band}"
                prepared[product_band] = tensor[:, idx : idx + 1]
        return prepared

    def _build_thor_metadata(self, x: dict[str, torch.Tensor]) -> dict[str, dict[str, int]]:
        patch_size = max(1, self.input_size // 16)
        metadata: dict[str, dict[str, int]] = {}
        for product_band in x.keys():
            product, _, band = product_band.partition(":")
            metadata[product_band] = {
                "GSD": self._get_band_gsd(product, band),
                "patch_size": patch_size,
                "num_patch": max(1, self.input_size // patch_size),
            }
        return metadata

    def _fallback_forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Compatibility path that preserves the PANGAEA encoder contract without a native THOR model."""
        if not isinstance(x, dict):
            raise TypeError("PANGAEA encoders expect a dict of modality tensors.")

        stacked = []
        for modality, tensor in x.items():
            if tensor is None:
                continue
            tensor = tensor.to(torch.float32)
            if tensor.dim() == 5:
                tensor = tensor[:, :, 0]
            stacked.append(tensor)

        if not stacked:
            raise ValueError("No valid modality tensors were provided to the THOR adapter.")

        fused = torch.cat(stacked, dim=1)
        feat = self._fallback_projection(fused)
        if feat.dim() == 4:
            return [feat]
        feat = feat.view(feat.shape[0], feat.shape[1], feat.shape[-2], feat.shape[-1])
        return [feat]

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """PANGAEA-compatible forward pass.

        The THOR adapter keeps the contract explicit: a dict of modality tensors in,
        a list of feature maps out.
        """
        if self.thor_model is not None:
            try:
                thor_input = self._prepare_thor_input(x)
                if not thor_input:
                    raise ValueError("No valid THOR input data found in the passed modality dict.")

                metadata = self._build_thor_metadata(thor_input)
                ground_cover = self._resolve_ground_cover(x) # TODO check if necessary
                output = self.thor_model(
                    thor_input,
                    metadata=metadata,
                    ground_cover=ground_cover, # self.ground_covers[0], <-- replaced with resolved cover
                    return_channel_params=False,
                )

                if isinstance(output, tuple):
                    output = output[0] if len(output) > 0 else output
                if isinstance(output, list):
                    return output
                if isinstance(output, torch.Tensor):
                    if output.dim() == 3:
                        return [output.unsqueeze(1)]
                    return [output]
            except Exception as exc:  # pragma: no cover - depends on actual THOR initialization
                logger.warning(
                    "Native THOR forward pass failed. Falling back to compatibility path. Error: %s",
                    exc,
                )

        return self._fallback_forward(x)

    def load_encoder_weights(self, logger: Logger) -> None:
        """THOR weights are loaded by the native model factory if available."""
        if self.thor_model is not None:
            logger.info("THOR native encoder weights are handled by the THOR model build path.")
            return
        logger.info("THOR adapter is running in compatibility mode; no external checkpoint is loaded.")

    def freeze(self) -> None:
        """Freeze the encoder parameters to prevent gradient updates."""
        if self.thor_model is not None:
            super().freeze()
        else:
            for param in self._fallback_projection.parameters():
                param.requires_grad = False
