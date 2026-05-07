"""
Empirical Margin Sensitivity (EMS) and Transformer Probing Core.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import logging

logger = logging.getLogger(__name__)

def get_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """Extract the core transformer backbone from a HF model wrapper."""
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    return model

def get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Retrieve the transformer layer stack."""
    backbone = get_backbone(model)
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "h"):
        return backbone.h
    if hasattr(backbone, "decoder") and hasattr(backbone.decoder, "layers"):
        return backbone.decoder.layers
    raise ValueError("Unsupported architecture: cannot identify transformer layers.")

def get_mlp_down_proj(layer: torch.nn.Module) -> torch.nn.Module:
    """Identify the MLP output projection (value-writing) component."""
    if hasattr(layer, "mlp"):
        if hasattr(layer.mlp, "down_proj"): return layer.mlp.down_proj
        if hasattr(layer.mlp, "fc2"): return layer.mlp.fc2
        if hasattr(layer.mlp, "c_proj"): return layer.mlp.c_proj
    if hasattr(layer, "feed_forward"):
        if hasattr(layer.feed_forward, "w2"): return layer.feed_forward.w2
    raise ValueError("Unsupported architecture: cannot identify MLP down-projection.")

class EMSProbe:
    """
    Core engine for mechanistic interventions on transformer MLP neurons.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layers = get_layers(model)
        
        sample_layer = self.layers[0]
        down_proj = get_mlp_down_proj(sample_layer)
        self.hidden_size = down_proj.weight.shape[0]
        self.intermediate_size = down_proj.weight.shape[1]

        self.num_heads = getattr(model.config, "num_attention_heads", None)
        if self.num_heads is None:
            text_cfg = getattr(model.config, "text_config", None)
            if text_cfg: self.num_heads = getattr(text_cfg, "num_attention_heads", None)

    def get_layer_weight(self, layer_idx: int) -> torch.Tensor:
        return get_mlp_down_proj(self.layers[layer_idx]).weight

    def _get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]

    def compute_taylor_scores(self, layer_idx: int, adv_pairs: List[Dict[str, Any]]) -> torch.Tensor:
        """Stage 0: Fast screening using gradient-based sensitivity."""
        original_requires_grad = {}
        for name, p in self.model.named_parameters():
            original_requires_grad[name] = p.requires_grad
            p.requires_grad = False

        target_weight = self.get_layer_weight(layer_idx)
        target_weight.requires_grad = True
        grad_accum = torch.zeros_like(target_weight, dtype=torch.float32)

        try:
            for pair in adv_pairs:
                self.model.zero_grad(set_to_none=True)
                logits = self._get_logits(pair["input_ids"].to(self.model.device), pair["attention_mask"].to(self.model.device))
                margin = logits[0, pair["pos_id"]] - logits[0, pair["neg_id"]]
                margin.backward()
                if target_weight.grad is not None:
                    grad_accum += target_weight.grad.detach().to(torch.float32)
        finally:
            for name, p in self.model.named_parameters():
                p.requires_grad = original_requires_grad[name]

        with torch.no_grad():
            w = target_weight.detach().to(torch.float32)
            scale = w.abs().mean(dim=0, keepdim=True)
            crushed = w.sign() * scale
            delta = crushed - w
            col_scores = torch.abs((grad_accum * delta).sum(dim=0))

        return col_scores

    def simulate_column_crush(self, layer_idx: int, col_idx: int) -> torch.Tensor:
        weight = self.get_layer_weight(layer_idx)
        original_col = weight[:, col_idx].clone()
        scale = original_col.abs().mean()
        crushed = original_col.sign() * scale
        with torch.no_grad():
            weight[:, col_idx] = crushed.to(weight.dtype)
        return original_col

    def restore_column(self, layer_idx: int, col_idx: int, original_col: torch.Tensor):
        weight = self.get_layer_weight(layer_idx)
        with torch.no_grad():
            weight[:, col_idx] = original_col.to(weight.dtype)

    def compute_margins(self, adv_pairs: List[Dict[str, Any]], batch_size: int = 1) -> torch.Tensor:
        margins = []
        with torch.no_grad():
            for pair in adv_pairs:
                logits = self._get_logits(pair["input_ids"].to(self.model.device), pair["attention_mask"].to(self.model.device))
                m = (logits[0, pair["pos_id"]] - logits[0, pair["neg_id"]]).item()
                margins.append(m)
        return torch.tensor(margins, dtype=torch.float32)
