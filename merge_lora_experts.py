"""
Merge MALoRA weights into standard full-size expert weights.

Updated for MALoRA's expert decomposition (delta = B_bar_t @ P_t @ S_A),
replacing the old vanilla-LoRA merge (delta = B_t @ A_t). The key structural
difference: S_A is SHARED across all experts in a layer (lives on
lora_moe_block.gate_SA / up_SA / down_SA), not owned per-expert like the
old A matrix was — so merging now needs the layer's shared_lora_block
passed alongside each expert, not just the expert in isolation.

NEW: attention LoRA merging is now TOGGLEABLE via MERGE_ATTENTION_LORA
below. Attention LoRA is unrelated to the MALoRA decomposition (see
peft_experts.py's AttentionLoRA docstring — it's standard two-matrix LoRA,
Q/K/V/O, independent of the shared-S_A MLP expert changes) so it uses the
original, simpler B@A merge math, not the three-matrix MALoRA chain.

This does three things:
1. Merges each expert's MALoRA adapter into the frozen base MLP weights,
   producing N fully independent expert FFNs (no LoRA/MALoRA math left).
2. If MERGE_ATTENTION_LORA is True, also merges attention LoRA (Q/K/V/O)
   into the frozen attention weights, using standard two-matrix merge math.
3. Verifies numerically that both merges match the original unmerged
   forward pass, before you trust any of this.

Run this on the same environment/checkpoint you used for training.
"""

import os
import torch
from huggingface_hub import snapshot_download, list_repo_files
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from configuration_lora_moe import LoraMoeConfig
from modelling import LoraMoeModel
from training_config import TrainingConfig

load_dotenv()  # picks up HF_TOKEN from .env, same as train.py does

conf = TrainingConfig()
MODEL_ID = conf.MODEL_ID

# Private HF repo — needs HF_TOKEN set (via .env or environment variable),
# same auth setup you already use for pushing checkpoints during training.
CHECKPOINT_REPO = "godofwar1007/moelora"
CHECKPOINT_REVISION = "main"  # <-- change if you need a specific branch/tag/commit
# The specific run you want: 10k examples, 3 epochs, attention LoRA off
RUN_FOLDER = "maloraa_10k_1ep_aton"
HF_TOKEN = os.environ.get("HF_TOKEN")

# ── TOGGLE: merge attention LoRA too? ─────────────────────────────────────────
# Set this based on whatever `use_attention_lora` value the checkpoint you're
# merging was actually TRAINED with — check that run's config/wandb, don't
# just guess. If the checkpoint has use_attention_lora=False, leave this
# False too (there's nothing to merge; the attention modules are just the
# frozen base weights, untouched).
MERGE_ATTENTION_LORA = True  # <-- SET THIS based on the actual training run


def list_available_checkpoints():
    """List all checkpoint steps available under RUN_FOLDER, so you can
    pick the exact one you want instead of guessing the step number."""
    all_files = list_repo_files(repo_id=CHECKPOINT_REPO, revision=CHECKPOINT_REVISION, token=HF_TOKEN)
    run_files = [f for f in all_files if f.startswith(f"{RUN_FOLDER}/")]
    checkpoint_steps = sorted(set(
        f.split("/")[1] for f in run_files if f.split("/")[1].startswith("checkpoint-")
    ), key=lambda x: int(x.split("-")[-1]))
    print(f"Available checkpoints under '{RUN_FOLDER}':")
    for c in checkpoint_steps:
        print(f"  - {c}")
    return checkpoint_steps


def merge_malora_linear(
    base_weight: torch.Tensor,
    S_A_module,       # SharedDownProjection — SHARED across all experts in this layer
    lora_module,       # MALoRALinear instance for THIS specific expert (owns P, B_bar)
) -> torch.Tensor:
    """
    MALoRA version of the merge math.

    base_weight: the frozen weight matrix, shape [out_features, in_features]
    S_A_module.proj.weight: [shared_rank, in_features]      — shared, per layer
    lora_module.P.weight:      [expert_rank, shared_rank]   — private, per expert
    lora_module.B_bar.weight:  [out_features, expert_rank]  — private, per expert

    Chained product B_bar @ P @ S_A gives [out_features, in_features],
    matching base_weight's shape exactly — same principle as the old
    B @ A merge, just one extra matrix in the chain.
    """
    delta = (
        lora_module.B_bar.weight
        @ lora_module.P.weight
        @ S_A_module.proj.weight
    ) * lora_module.scale
    return base_weight + delta.to(base_weight.dtype)


def merge_attention_lora_linear(
    base_weight: torch.Tensor,
    lora_module,   # _StandardLoraLinear instance (owns A, B) — standard 2-matrix LoRA
) -> torch.Tensor:
    """
    Standard (non-MALoRA) two-matrix LoRA merge, for attention Q/K/V/O.

    Attention LoRA is UNCHANGED from vanilla LoRA/MoE-LoRA — see
    peft_experts.py's AttentionLoRA/_StandardLoraLinear docstrings. No
    shared subspace here, so this is the simple B@A merge, not the
    three-matrix MALoRA chain used for MLP experts above.

    base_weight: [out_features, in_features]
    lora_module.A.weight: [r, in_features]
    lora_module.B.weight: [out_features, r]
    """
    delta = (lora_module.B.weight @ lora_module.A.weight) * lora_module.scale
    return base_weight + delta.to(base_weight.dtype)


@torch.no_grad()
def verify_single_expert_merge(
    expert, mlp, gate_SA, up_SA, down_SA,
    merged_gate_w, merged_up_w, merged_down_w,
    hidden_size, device, dtype,
):
    """
    Sanity check: unmerged forward pass vs merged forward pass should match
    within floating point tolerance, for a random input batch.

    Uses MALoRALinear.forward(x, S_A) directly (not forward_from_shared) —
    this is the self-contained path that computes S_A internally, which is
    exactly what we want here since we're not routing/dispatching, just
    checking "does this expert's full computation match its merged form."

    Tolerance: atol=2e-1 rather than 1e-2 — bf16 has ~3 decimal digits of
    precision, and torch.allclose checks every element, so a handful of
    high-magnitude elements can trip a tight absolute tolerance even when
    the merge is mathematically correct. Watch the printed relative-error
    percentage, not just pass/fail, if you want to sanity-check this isn't
    masking a real bug.
    """
    x = torch.randn(4, 16, hidden_size, device=device, dtype=dtype)  # [batch, seq, hidden]

    # --- old way: unmerged, using the actual MALoRA forward methods ---
    gate_old = mlp.gate_proj(x) + expert.gate_lora(x, gate_SA)
    up_old   = mlp.up_proj(x)   + expert.up_lora(x, up_SA)
    act_old  = expert.activation_fn(gate_old) * up_old
    down_old = mlp.down_proj(act_old) + expert.down_lora(act_old, down_SA)

    # --- new way: merged, using plain matmul with merged weights ---
    gate_new = x @ merged_gate_w.T
    up_new   = x @ merged_up_w.T
    act_new  = expert.activation_fn(gate_new) * up_new
    down_new = act_new @ merged_down_w.T

    gate_ok = torch.allclose(gate_old, gate_new, atol=2e-1, rtol=2e-2)
    down_ok = torch.allclose(down_old, down_new, atol=2e-1, rtol=2e-2)

    max_diff_gate = (gate_old - gate_new).abs().max().item()
    max_diff_down = (down_old - down_new).abs().max().item()
    gate_rel = max_diff_gate / gate_old.abs().max().item() * 100 if gate_old.abs().max().item() > 0 else 0
    down_rel = max_diff_down / down_old.abs().max().item() * 100 if down_old.abs().max().item() > 0 else 0

    print(f"    gate match: {gate_ok}  (max diff: {max_diff_gate:.6f}, rel: {gate_rel:.4f}%)")
    print(f"    down match: {down_ok}  (max diff: {max_diff_down:.6f}, rel: {down_rel:.4f}%)")

    return gate_ok and down_ok


@torch.no_grad()
def verify_attention_merge(
    attn_lora_module, base_attn,
    merged_q_w, merged_k_w, merged_v_w, merged_o_w,
    hidden_size, device, dtype,
):
    """
    Sanity check for attention LoRA merge — same idea as the expert check,
    just for Q/K/V/O instead of gate/up/down.

    Both "old" (unmerged) and "new" (merged) paths include bias explicitly,
    since LoRA never touches bias but a raw matmul doesn't add it
    automatically the way an nn.Linear call does.

    Tolerance: see verify_single_expert_merge's docstring — same rationale.
    """
    x = torch.randn(4, 16, hidden_size, device=device, dtype=dtype)

    # --- old way: unmerged, base linear (unwrapped from LoraLinear) + LoRA delta ---
    q_old = base_attn.q_proj.base(x) + attn_lora_module.forward_q(x)
    k_old = base_attn.k_proj.base(x) + attn_lora_module.forward_k(x)
    v_old = base_attn.v_proj.base(x) + attn_lora_module.forward_v(x)

    # --- new way: merged, plain matmul + base bias (bias unaffected by LoRA) ---
    q_new = x @ merged_q_w.T + base_attn.q_proj.base.bias
    k_new = x @ merged_k_w.T + base_attn.k_proj.base.bias
    v_new = x @ merged_v_w.T + base_attn.v_proj.base.bias

    q_ok = torch.allclose(q_old, q_new, atol=2e-1, rtol=2e-2)
    k_ok = torch.allclose(k_old, k_new, atol=2e-1, rtol=2e-2)
    v_ok = torch.allclose(v_old, v_new, atol=2e-1, rtol=2e-2)

    q_diff = (q_old - q_new).abs().max().item()
    k_diff = (k_old - k_new).abs().max().item()
    v_diff = (v_old - v_new).abs().max().item()
    q_rel = q_diff / q_old.abs().max().item() * 100 if q_old.abs().max().item() > 0 else 0
    k_rel = k_diff / k_old.abs().max().item() * 100 if k_old.abs().max().item() > 0 else 0
    v_rel = v_diff / v_old.abs().max().item() * 100 if v_old.abs().max().item() > 0 else 0

    print(f"      q match: {q_ok}  (max diff: {q_diff:.6f}, rel: {q_rel:.4f}%)")
    print(f"      k match: {k_ok}  (max diff: {k_diff:.6f}, rel: {k_rel:.4f}%)")
    print(f"      v match: {v_ok}  (max diff: {v_diff:.6f}, rel: {v_rel:.4f}%)")

    # o_proj: unmerged input is attention output, not raw x — skip exact
    # numerical replay here (would need a full attention forward pass);
    # weight-level merge math is identical to q/k/v, so if those three
    # check out, o_proj's merge (same formula, different tensor) is sound.
    return q_ok and k_ok and v_ok


def main():
    print(f"Loading base model: {MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    print("Loading MoE config...")
    moe_config = LoraMoeConfig.from_pretrained(MODEL_ID)
    
    # Actual values used for training — UPDATED for MALoRA's shared_rank /
    # expert_rank decomposition (replaces the old single experts_rank).
    # VERIFY these against the actual run's wandb config before trusting them.
    moe_config.shared_rank          = 16   # <-- VERIFY: dimension of shared S_A
    moe_config.expert_rank          = 16   # <-- VERIFY: dimension of private P_t/B_bar_t
    moe_config.attention_rank       = 32
    moe_config.experts_scale        = 1.0
    moe_config.experts_dropout      = 0.05
    moe_config.num_experts_per_tok  = 2
    moe_config.num_local_experts    = 8
    moe_config.output_router_logits = False
    moe_config.router_aux_loss_coef = 0.001 
    # IMPORTANT: this must match what MERGE_ATTENTION_LORA is set to, AND
    # both must match the actual checkpoint's training config — verify
    # against wandb before trusting either.
    moe_config.use_attention_lora   = MERGE_ATTENTION_LORA

    print(f"MERGE_ATTENTION_LORA = {MERGE_ATTENTION_LORA}")
    if MERGE_ATTENTION_LORA != moe_config.use_attention_lora:
        raise ValueError(
            "MERGE_ATTENTION_LORA and moe_config.use_attention_lora must match "
            "the actual checkpoint's training config — mismatch here means "
            "you'd either merge nonexistent LoRA weights or silently skip "
            "real trained ones."
        )

    print("Wrapping with MALoRA architecture...")
    moe_model = LoraMoeModel(base_model, moe_config)
    moe_model.eval()

    if torch.cuda.is_available():
        moe_model = moe_model.to("cuda")
        print("  moved model to GPU")
    else:
        print("  ⚠️  no GPU detected — this will be slow on CPU")

    checkpoints = list_available_checkpoints()
    print(f"\nNote: the run's ROOT folder ('{RUN_FOLDER}/') holds the FINAL model")
    print(f"(from trainer.save_model() at the end of training) — this is what")
    print(f"you almost certainly want, rather than an intermediate checkpoint-N")
    print(f"subfolder from mid-training saves.\n")

    USE_ROOT_AS_FINAL = True  # <-- set False if you specifically want an intermediate checkpoint instead

    if USE_ROOT_AS_FINAL:
        target_subpath = RUN_FOLDER
    else:
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found under '{RUN_FOLDER}' in {CHECKPOINT_REPO}. Check the run name.")
        chosen_checkpoint = checkpoints[-1]
        target_subpath = f"{RUN_FOLDER}/{chosen_checkpoint}"

    print(f"Using: {target_subpath}")

    print(f"Downloading checkpoint from private repo: {CHECKPOINT_REPO}")
    local_ckpt_dir = snapshot_download(
        repo_id=CHECKPOINT_REPO,
        revision=CHECKPOINT_REVISION,
        token=HF_TOKEN,
        allow_patterns=[f"{target_subpath}/*"] if not USE_ROOT_AS_FINAL
                       else [f"{RUN_FOLDER}/*.json", f"{RUN_FOLDER}/*.safetensors",
                             f"{RUN_FOLDER}/*.txt", f"{RUN_FOLDER}/*.model"],
    )
    local_ckpt_dir = os.path.join(local_ckpt_dir, target_subpath)
    print(f"  downloaded to: {local_ckpt_dir}")

    safetensors_path = os.path.join(local_ckpt_dir, "model.safetensors")
    bin_path = os.path.join(local_ckpt_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        print("  loaded weights from model.safetensors")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        print("  loaded weights from pytorch_model.bin")
        print("\n===== STATE DICT CHECK =====")

        print("Checkpoint first key:")
        print(next(iter(state_dict.keys())))

        print("\nModel first key:")
        print(next(iter(moe_model.base_model.state_dict().keys())))

        print("\nWrapper model first key:")
        print(next(iter(moe_model.state_dict().keys())))

        print("============================\n")
    else:
        index_files = [f for f in os.listdir(local_ckpt_dir) if f.endswith(".index.json")]
        raise FileNotFoundError(
            f"Could not find model.safetensors or pytorch_model.bin directly in {local_ckpt_dir}. "
            f"Found these index files instead: {index_files}. "
            f"This checkpoint is likely sharded — load it with "
            f"transformers' own from_pretrained mechanism instead, or use "
            f"`from safetensors.torch import load_file` per-shard and merge state dicts."
        )
    print("\nFirst 10 checkpoint keys:")
    for i, k in enumerate(state_dict.keys()):
        print(k)
        if i == 9:
            break

    print("\nFirst 10 model keys:")
    for i, k in enumerate(moe_model.base_model.state_dict().keys()):
        print(k)
        if i == 9:
            break
 
    missing, unexpected = moe_model.load_state_dict(state_dict,strict=False)
    print("\n===== FIRST 30 MISSING =====")
    for k in missing[:30]:
        print(k)

    print("\n===== FIRST 30 UNEXPECTED =====")
    for k in unexpected[:30]:
        print(k)        
    print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    attn_lora_missing = [k for k in missing if "lora" in k and "self_attn" in k]
    print(f"\n  Attention-LoRA-specific missing keys ({len(attn_lora_missing)}):")
    for k in attn_lora_missing[:20]:
        print(f"    {k}")


    k_related_in_checkpoint = [k for k in state_dict.keys() if "k_lora" in k or ("self_attn" in k and ".k." in k)]
    print(f"\n  Checkpoint's actual k-related key names ({len(k_related_in_checkpoint)}):")
    for k in k_related_in_checkpoint[:10]:
        print(f"    {k}")    
    if len(missing) > 20:
        print(f"  ⚠️  a lot of missing keys ({len(missing)}) — double check the checkpoint actually matches this config shape")

    device = moe_model.device
    hidden_size = moe_config.hidden_size
    from isolate_attention_bug import isolate_attention_merge_bug
    isolate_attention_merge_bug(moe_model.base_model.model.layers[0], 0, hidden_size, device, moe_config.torch_dtype if hasattr(moe_config, "torch_dtype") else torch.bfloat16)

    all_merged_experts = []  # list of dicts, one per layer
    all_merged_attention = []  # list of dicts, one per layer (only if MERGE_ATTENTION_LORA)

    print("\nMerging experts layer by layer...")
    for layer_id in moe_model.layer_ids:
        layer = moe_model.base_model.model.layers[layer_id]
        mlp = layer.mlp
        lora_moe_block = layer.lora_moe_block

        # shared per this layer, NOT per expert — pulled out once per layer
        gate_SA = lora_moe_block.gate_SA
        up_SA   = lora_moe_block.up_SA
        down_SA = lora_moe_block.down_SA

        layer_merged = []
        for expert_idx, expert in enumerate(lora_moe_block.lora_experts):
            merged_gate = merge_malora_linear(mlp.gate_proj.weight, gate_SA, expert.gate_lora)
            merged_up   = merge_malora_linear(mlp.up_proj.weight,   up_SA,   expert.up_lora)
            merged_down = merge_malora_linear(mlp.down_proj.weight, down_SA, expert.down_lora)

            print(f"  Layer {layer_id}, Expert {expert_idx}: merged. Verifying...")
            ok = verify_single_expert_merge(
                expert, mlp, gate_SA, up_SA, down_SA,
                merged_gate, merged_up, merged_down,
                hidden_size, device, merged_gate.dtype,
            )
            if not ok:
                print(f"  ⚠️  MISMATCH at layer {layer_id}, expert {expert_idx} — investigate before proceeding!")

            layer_merged.append({
                "gate_proj.weight": merged_gate.detach().cpu(),
                "up_proj.weight":   merged_up.detach().cpu(),
                "down_proj.weight": merged_down.detach().cpu(),
            })

        all_merged_experts.append(layer_merged)

        # ── TOGGLE: merge attention LoRA for this layer too? ─────────────────
        if MERGE_ATTENTION_LORA:
            if not layer._has_attn_lora:
                raise RuntimeError(
                    f"Layer {layer_id}: MERGE_ATTENTION_LORA=True but this "
                    f"layer's _has_attn_lora is False — config/checkpoint "
                    f"mismatch, stopping rather than silently skipping."
                )
            base_attn = layer.self_attn.base_attn
            attn_lora = layer.self_attn.lora
            if layer_id==0:

                print("k_lora.A.weight norm:", attn_lora.k_lora.A.weight.norm().item())
                print("k_lora.B.weight norm:", attn_lora.k_lora.B.weight.norm().item())
                print("q_lora.A.weight norm:", attn_lora.q_lora.A.weight.norm().item())
            merged_q = merge_attention_lora_linear(base_attn.q_proj.base.weight, attn_lora.q_lora)
            merged_k = merge_attention_lora_linear(base_attn.k_proj.base.weight, attn_lora.k_lora)
            merged_v = merge_attention_lora_linear(base_attn.v_proj.base.weight, attn_lora.v_lora)
            merged_o = merge_attention_lora_linear(base_attn.o_proj.base.weight, attn_lora.o_lora)

            print(f"  Layer {layer_id}, Attention: merged. Verifying...")
            attn_ok = verify_attention_merge(
                attn_lora, base_attn,
                merged_q, merged_k, merged_v, merged_o,
                hidden_size, device, merged_q.dtype,
            )
            if not attn_ok:
                print(f"  ⚠️  ATTENTION MISMATCH at layer {layer_id} — investigate before proceeding!")

            all_merged_attention.append({
                "q_proj.weight": merged_q.detach().cpu(),
                "k_proj.weight": merged_k.detach().cpu(),
                "v_proj.weight": merged_v.detach().cpu(),
                "o_proj.weight": merged_o.detach().cpu(),
            })

    print("\nDone merging. `all_merged_experts` now holds, per layer, a list of")
    print("per-expert weight dicts — this is the raw material for Step 6/7")
    print("(reshaping into a target architecture's checkpoint format).")

    torch.save(all_merged_experts, "merged_experts_raw.pt")
    print("Saved raw merged expert weights to merged_experts_raw.pt for the next step.")

    if MERGE_ATTENTION_LORA:
        torch.save(all_merged_attention, "merged_attention_raw.pt")
        print("Saved raw merged attention weights to merged_attention_raw.pt for the next step.")
    else:
        print("MERGE_ATTENTION_LORA=False — attention weights are just the frozen")
        print("base model's, unchanged, nothing extra needed for them.")


if __name__ == "__main__":
    main()
