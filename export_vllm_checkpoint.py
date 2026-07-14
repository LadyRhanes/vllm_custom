"""
Export a full vLLM-loadable checkpoint from the merged MALoRA weights.

This is Step 6/7 that merge_lora_experts.py's final print statement refers
to — it takes:
  1. The merged expert weights (already computed in-memory during merging —
     this script is meant to run as a continuation of merge_lora_experts.py's
     main(), not as a standalone reload of the .pt files)
  2. The merged attention weights (if MERGE_ATTENTION_LORA was True)
  3. Everything that was NEVER touched by merging at all: embed_tokens,
     lm_head, final norm, per-layer input/post_attention layernorms, and
     the ROUTER's gate weight (router.gate is trained but never merged —
     it's not a LoRA/MALoRA delta, it's used as-is)

...and writes one real HF-format checkpoint (config.json + model.safetensors)
with vLLM-expected key names, ready for:

    from vllm import ModelRegistry
    ModelRegistry.register_model("LoraMoeForCausalLM", "lora_moe_vllm:LoraMoeForCausalLMVLLM")
    llm = LLM(model="vllm_export_checkpoint/", ...)

WEIGHT NAMING — matches lora_moe_vllm.py's module hierarchy exactly:
    model.embed_tokens.weight
    model.norm.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.self_attn.q_proj.weight / .k_proj.weight / .v_proj.weight / .o_proj.weight
        (UNFUSED — hf_to_vllm_mapper's WeightsMapper fuses these into
        qkv_proj automatically via AutoWeightsLoader, so export unfused)
    model.layers.{i}.lora_moe_block.gate.weight
        (this is router.gate.weight from the ORIGINAL model — never merged)
    model.layers.{i}.lora_moe_block.experts.{e}.gate_proj.weight / .up_proj.weight / .down_proj.weight
    lm_head.weight

KNOWN RISK (matches lora_moe_vllm.py's own TODO #1): FusedMoE often needs
weights loaded via an explicit expert_params_mapping (FusedMoE.make_expert_params_mapping)
rather than picked up automatically by generic AutoWeightsLoader — this
naming convention is the right ONE to target either way, but load_weights()
in lora_moe_vllm.py may need updating to use that explicit mapping if a
plain AutoWeightsLoader pass doesn't correctly stack the per-expert tensors.
Flagging here so it's not a surprise; test and see which is true before
assuming either way.
"""

import json
import os
import torch
from safetensors.torch import save_file


def export_vllm_checkpoint(
    moe_model,
    all_merged_experts,
    all_merged_attention,  # None if MERGE_ATTENTION_LORA was False
    output_dir: str = "vllm_export_checkpoint",
):
    """
    Call this at the END of merge_lora_experts.py's main(), after merging
    is done — moe_model, all_merged_experts, and all_merged_attention are
    all already sitting in memory at that point, no need to reload anything
    from disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    state_dict = {}

    base_model = moe_model.base_model  # the Qwen2ForCausalLM-shaped wrapper
    config = moe_model.config

    print(f"\nExporting vLLM checkpoint to {output_dir}/ ...")

    # ── untouched, model-level weights ────────────────────────────────────
    state_dict["model.embed_tokens.weight"] = base_model.model.embed_tokens.weight.detach().cpu()
    state_dict["model.norm.weight"] = base_model.model.norm.weight.detach().cpu()
    state_dict["lm_head.weight"] = base_model.lm_head.weight.detach().cpu()

    # ── per-layer weights ──────────────────────────────────────────────────
    for layer_id in moe_model.layer_ids:
        layer = base_model.model.layers[layer_id]
        prefix = f"model.layers.{layer_id}"

        # layernorms — untouched by any merge
        state_dict[f"{prefix}.input_layernorm.weight"] = \
            layer.input_layernorm.weight.detach().cpu()
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = \
            layer.post_attention_layernorm.weight.detach().cpu()

        # attention — merged if MERGE_ATTENTION_LORA was True, else base weights as-is
        if all_merged_attention is not None:
            attn_weights = all_merged_attention[layer_id]
            state_dict[f"{prefix}.self_attn.q_proj.weight"] = attn_weights["q_proj.weight"]
            state_dict[f"{prefix}.self_attn.k_proj.weight"] = attn_weights["k_proj.weight"]
            state_dict[f"{prefix}.self_attn.v_proj.weight"] = attn_weights["v_proj.weight"]
            state_dict[f"{prefix}.self_attn.o_proj.weight"] = attn_weights["o_proj.weight"]
            base_attn = layer.self_attn.base_attn
            state_dict[f"{prefix}.self_attn.q_proj.bias"] = base_attn.q_proj.base.bias.detach().cpu()
            state_dict[f"{prefix}.self_attn.k_proj.bias"] = base_attn.k_proj.base.bias.detach().cpu()
            state_dict[f"{prefix}.self_attn.v_proj.bias"] = base_attn.v_proj.base.bias.detach().cpu()

        else:
            base_attn = layer.self_attn.base_attn if layer._has_attn_lora else layer.self_attn
            state_dict[f"{prefix}.self_attn.q_proj.weight"] = base_attn.q_proj.weight.detach().cpu()
            state_dict[f"{prefix}.self_attn.k_proj.weight"] = base_attn.k_proj.weight.detach().cpu()
            state_dict[f"{prefix}.self_attn.v_proj.weight"] = base_attn.v_proj.weight.detach().cpu()
            state_dict[f"{prefix}.self_attn.o_proj.weight"] = base_attn.o_proj.weight.detach().cpu()
            state_dict[f"{prefix}.self_attn.q_proj.bias"] = base_attn.q_proj.bias.detach().cpu()
            state_dict[f"{prefix}.self_attn.k_proj.bias"] = base_attn.k_proj.bias.detach().cpu()
            state_dict[f"{prefix}.self_attn.v_proj.bias"] = base_attn.v_proj.bias.detach().cpu()

        # router — trained, but NEVER merged (it's not a LoRA delta, it's
        # used as-is). Pull straight from the original in-memory model.
        router_gate_weight = layer.lora_moe_block.router.gate.weight.detach().cpu()
        state_dict[f"{prefix}.lora_moe_block.gate.weight"] = router_gate_weight

        # experts — already merged, just need renaming into this layout
        layer_experts = all_merged_experts[layer_id]
        for expert_idx, expert_weights in enumerate(layer_experts):
            expert_prefix = f"{prefix}.lora_moe_block.experts.{expert_idx}"
            state_dict[f"{expert_prefix}.gate_proj.weight"] = expert_weights["gate_proj.weight"]
            state_dict[f"{expert_prefix}.up_proj.weight"] = expert_weights["up_proj.weight"]
            state_dict[f"{expert_prefix}.down_proj.weight"] = expert_weights["down_proj.weight"]

        print(f"  layer {layer_id}: exported ({len(layer_experts)} experts)")

    # ── save weights ────────────────────────────────────────────────────────
    # ensure everything is contiguous bf16 — safetensors requires contiguous
    # tensors, and merged weights from matmul chains aren't guaranteed to be
    state_dict = {k: v.contiguous().to(torch.bfloat16) for k, v in state_dict.items()}
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    print(f"  saved model.safetensors ({len(state_dict)} tensors)")

    # ── save config ────────────────────────────────────────────────────────
    # architectures MUST be exactly this string to match
    # ModelRegistry.register_model("LoraMoeForCausalLM", ...) — same
    # naming-consistency issue that broke the Transformers-backend fallback
    # earlier; getting it right here from the start.
    config_dict = config.to_dict()
    config_dict["architectures"] = ["LoraMoeForCausalLM"]
    # not using trust_remote_code path this time (custom-registered native
    # architecture instead), so auto_map isn't needed for loading — but
    # left in place if present, harmless either way.
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  saved config.json (architectures={config_dict['architectures']})")

    print(f"\n✅ Export complete: {output_dir}/")
    print(f"   Total parameters exported: {sum(v.numel() for v in state_dict.values()) / 1e6:.1f}M")


# ── If running standalone (not appended to merge_lora_experts.py) ────────────
# Reloads merged_experts_raw.pt / merged_attention_raw.pt from disk and
# rebuilds moe_model fresh to get the untouched weights. Slower (reloads
# and re-wraps the base model) but useful if you already ran the merge step
# in a previous session and don't want to redo it just to export.
if __name__ == "__main__":
    import torch as _torch
    from configuration_lora_moe import LoraMoeConfig
    from modelling import LoraMoeModel
    from training_config import TrainingConfig
    from transformers import AutoModelForCausalLM

    conf = TrainingConfig()

    print("Standalone export mode: reloading merged weights from disk...")
    all_merged_experts = _torch.load("merged_experts_raw.pt")

    all_merged_attention = None
    if os.path.exists("merged_attention_raw.pt"):
        all_merged_attention = _torch.load("merged_attention_raw.pt")
        print("  found merged_attention_raw.pt — attention was merged, including it")
    else:
        print("  no merged_attention_raw.pt found — attention stays as base weights")

    print("Rebuilding moe_model to recover untouched weights (embed/norm/layernorms/router)...")
    print("⚠️  IMPORTANT: the config values below MUST match what was used during merging —")
    print("   this is NOT re-verified automatically. Check against merge_lora_experts.py's")
    print("   own config block before trusting this standalone path.")
    base_model = AutoModelForCausalLM.from_pretrained(
        conf.MODEL_ID, torch_dtype=_torch.bfloat16, trust_remote_code=True
    )
    moe_config = LoraMoeConfig.from_pretrained(conf.MODEL_ID)
    moe_config.shared_rank = 16
    moe_config.expert_rank = 16
    moe_config.attention_rank = 32
    moe_config.num_experts_per_tok = 2
    moe_config.num_local_experts = 8
    moe_config.use_attention_lora = all_merged_attention is not None
    moe_model = LoraMoeModel(base_model, moe_config)

    print("⚠️  NOTE: this rebuild has RANDOM router/expert weights — only")
    print("    embed_tokens/norm/layernorms/router.gate get used from it below.")
    print("    If you need the router.gate weight specifically, this standalone")
    print("    path does NOT have the actual trained router weights loaded —")
    print("    load the real checkpoint's state_dict into moe_model first, same")
    print("    as merge_lora_experts.py's main() does, before calling this.")

    export_vllm_checkpoint(moe_model, all_merged_experts, all_merged_attention)
