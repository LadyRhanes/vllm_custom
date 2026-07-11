"""
Custom vLLM model registration for the merged MALoRA/MoE-LoRA architecture.

Adapted DIRECTLY from vLLM's real qwen2_moe.py source (not hand-built from
scratch) — this is a much better starting point than adapting Mixtral,
since Qwen2MoeAttention/Qwen2MoeDecoderLayer already have the EXACT
attention shape our model uses (GQA + QKV bias, standard RoPE) with zero
adaptation needed. Only the MoE block itself needed rewriting, since our
architecture is simpler than Qwen2MoE's (no shared_expert, no
mlp_only_layers — every layer is MoE, always).

IMPORT NOTE: the relative imports (`from .interfaces import ...`,
`from .utils import ...`) that this file originally had assumed it would
live inside vllm/model_executor/models/ alongside vLLM's own source. Since
this is a flat standalone file instead, those are rewritten below as
absolute imports (`from vllm.model_executor.models.interfaces import ...`)
so it imports correctly no matter where it sits.

STATUS: untested skeleton — written from real vLLM source, but not run
against actual vLLM internals yet. Test the moment compute is available;
every TODO below is something to verify, not something already confirmed.

Register with:
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "LoraMoeForCausalLM", "lora_moe_vllm:LoraMoeForCausalLMVLLM"
    )
"""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

# CHANGED: absolute imports instead of relative (.interfaces / .utils),
# since this file lives standalone rather than inside
# vllm/model_executor/models/. If vLLM's own internal module layout ever
# moves these two files, these two lines are the first thing to check.
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# ── MoE Block — REWRITTEN for our simpler architecture ───────────────────────
# Qwen2MoeSparseMoeBlock has a shared_expert + shared_expert_gate that our
# architecture doesn't have (MALoRA/MoE-LoRA never had a "shared expert"
# concept — that's a different Qwen2MoE-specific design). Dropped entirely
# here rather than carried over unused.
class LoraMoeBlockVLLM(nn.Module):
    """
    TODO when testing: confirm config.num_local_experts / num_experts_per_tok
    field names exactly match LoraMoeConfig (they should — verified against
    configuration_lora_moe.py at write time, but confirm against the actual
    merged checkpoint's config.json, not just the source file).
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # TODO when testing: our router also has a noise_gate
        # (lora_moe_block.router.noise_gate) used only during TRAINING —
        # at inference (eval), it contributes nothing (see
        # DispatchMoERouter.forward: `if self.training: noise = ...`), so
        # it's deliberately NOT loaded/used here. Confirm this by comparing
        # a served output against the HF reference for the same prompt,
        # not by assuming it's safe.
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,  # our merged experts use
                                                          # the SAME intermediate_size
                                                          # as the base MLP — NOT a
                                                          # separate moe_intermediate_size
                                                          # field like Qwen2MoE/Qwen3MoE
                                                          # have. TODO: confirm this
                                                          # against the actual merged
                                                          # expert weight shapes.
            renormalize=True,  # matches DispatchMoERouter's
                               # `top_k_weights / top_k_weights.sum(...)` normalization
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(orig_shape)


# ── Attention — copied near-verbatim from Qwen2MoeAttention ──────────────────
# This is EXACTLY the attention style our base model (Qwen2.5-Coder) uses —
# GQA with QKV bias, standard RoPE. No adaptation needed here, which is
# precisely why starting from qwen2_moe.py instead of mixtral.py was the
# right call.
class LoraMoeAttentionVLLM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 32768,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,  # Qwen2-style — matches our base model exactly
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LoraMoeDecoderLayerVLLM(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.self_attn = LoraMoeAttentionVLLM(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=getattr(config, "rope_parameters", None)
                or {"rope_theta": getattr(config, "rope_theta", 1000000.0)},
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # Every layer is MoE, always — no mlp_only_layers, no
        # decoder_sparse_step gating like Qwen2MoE has. Simpler than the
        # source this was adapted from, on purpose.
        self.lora_moe_block = LoraMoeBlockVLLM(
            config=config, quant_config=quant_config, prefix=f"{prefix}.lora_moe_block"
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.lora_moe_block(hidden_states)
        return hidden_states, residual


@support_torch_compile
class LoraMoeModelVLLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LoraMoeDecoderLayerVLLM(
                config=config, cache_config=cache_config,
                quant_config=quant_config, prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id) — same helper
        # qwen2_moe.py uses, pointed at our field names. Our merged
        # experts keep gate/up/down as three separate weights (not a
        # fused gate_up_proj), matching ckpt_gate_proj_name/ckpt_up_proj_name
        # being passed separately here.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Explicit weight-loading loop — REPLACES reliance on WeightsMapper's
        orig_to_new_stacked (removed in this vLLM version's WeightsMapper;
        confirmed by reading the installed vllm/model_executor/models/utils.py
        directly, not assumed). Pattern copied from the actual current
        Qwen2MoeModel.load_weights, adapted to our simpler architecture (no
        gate_up_proj fusion, no shared_expert, every layer is MoE).

        AutoWeightsLoader (called from LoraMoeForCausalLMVLLM.load_weights
        below) automatically delegates to a submodule's own load_weights()
        method when one is defined — this is why this method exists on the
        inner model class specifically, not just the outer ForCausalLM class.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # NOTE: no gate_up_proj entry — our merged experts keep gate/up
            # separate (see LoraMoeBlockVLLM's FusedMoE construction),
            # unlike Qwen2MoE which fuses them. Handled entirely via
            # expert_params_mapping below instead.
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "lora_moe_block.experts" in name:
                    # experts handled below via expert_params_mapping —
                    # skip here so name doesn't get mangled first
                    continue
                name = name.replace(weight_name, param_name)
                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LoraMoeForCausalLMVLLM(nn.Module, SupportsPP):
    """
    Top-level vLLM model for the merged MALoRA/MoE-LoRA architecture.

    NOTE ON WEIGHT LOADING: qkv stacking and per-expert weight mapping
    happen inside LoraMoeModelVLLM.load_weights (an explicit loop), not via
    a WeightsMapper on this class — see that method's docstring for why.
    gate_proj/up_proj are NOT fused into a single gate_up_proj (unlike
    Qwen2MoeForCausalLM's mapper) because our merged experts keep them as
    separate tensors.

    TODO when testing, in priority order (UPDATED after reading the actual
    installed vLLM source directly):
    1. RESOLVED: hf_to_vllm_mapper's WeightsMapper(orig_to_new_stacked=...)
       doesn't exist in this vLLM version — confirmed by extracting and
       reading vllm/model_executor/models/utils.py directly. Replaced with
       an explicit stacked_params_mapping + expert_params_mapping loop in
       LoraMoeModelVLLM.load_weights, copied from the actual current
       Qwen2MoeModel.load_weights pattern. AutoWeightsLoader automatically
       delegates to a submodule's own load_weights() when present, which is
       why this lives on the inner model, not just the outer class.
    2. Confirm config.intermediate_size (not a separate moe_intermediate_size)
       is the correct dimension for merged expert FFNs.
    3. Confirm tie_word_embeddings matches the merged checkpoint (original
       merge explicitly untied it — check config.json's actual value).
    """

    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = LoraMoeModelVLLM(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # No mapper argument — the actual stacking/expert-mapping logic
        # lives in self.model.load_weights (LoraMoeModelVLLM), which
        # AutoWeightsLoader calls automatically since that submodule
        # defines its own load_weights(). Matches the confirmed current
        # pattern in vllm/model_executor/models/qwen2_moe.py exactly.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
