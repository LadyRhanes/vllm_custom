"""
Test loading the merged MALoRA checkpoint via vLLM's NATIVE custom-model
registration (Path B) — no trust_remote_code, no Transformers-backend
compatibility gate to fight with. This registers our own vLLM model class
directly with vLLM's architecture registry, the same mechanism vLLM uses
internally for Mixtral/Qwen2MoE.

Run this in the SEPARATE vLLM environment (requirements-vllm.txt), pointed
at the output of export_vllm_checkpoint.py.
"""

CHECKPOINT_PATH = "vllm_export_checkpoint"  # <-- output dir from export_vllm_checkpoint.py

def main():
    print("Registering LoraMoeForCausalLM with vLLM's model registry...")
    try:
        from vllm import ModelRegistry
        # Adjust "lora_moe_vllm" if you saved the model file under a
        # different module name — must be importable (same directory, or
        # on PYTHONPATH).
        ModelRegistry.register_model(
            "LoraMoeForCausalLM", "lora_moe_vllm:LoraMoeForCausalLMVLLM"
        )
        print("✅ Registered successfully.")
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        print("\nThis means lora_moe_vllm.py itself has an import-time error —")
        print("check that all the vllm.model_executor.* imports at the top of")
        print("that file actually exist in your installed vLLM version (these")
        print("move around between versions more than most of the transformers")
        print("API does). Fix import errors here before going further.")
        return

    print(f"\nAttempting to load {CHECKPOINT_PATH} via vLLM's native path...")
    try:
        from vllm import LLM

        llm = LLM(
            model=CHECKPOINT_PATH,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            # NOTE: no trust_remote_code needed — this is vLLM's own
            # registered architecture, not the Transformers-backend fallback
            # that gave us trouble earlier.
        )
    except Exception as e:
        print(f"❌ vLLM failed to load the model:\n{e}")
        print("\nMost likely failure points, in order of probability:")
        print("  1. FusedMoE weight loading — if the error mentions shape")
        print("     mismatches or missing expert weights, load_weights() in")
        print("     lora_moe_vllm.py likely needs an explicit")
        print("     FusedMoE.make_expert_params_mapping() loop instead of the")
        print("     current plain AutoWeightsLoader pass — this was flagged")
        print("     as the single highest-risk assumption when the file was")
        print("     written.")
        print("  2. config.json missing a field lora_moe_vllm.py's __init__")
        print("     methods expect (check the actual AttributeError/KeyError")
        print("     text for the exact field name).")
        print("  3. qkv_proj fusion — if q/k/v shapes don't line up, double")
        print("     check GQA head counts (num_attention_heads vs")
        print("     num_key_value_heads) in the exported config.json match")
        print("     what LoraMoeAttentionVLLM expects.")
        return

    print("✅ Model loaded successfully via native vLLM registration!")

    print("\nRunning a quick generation test...")
    output = llm.generate(["def fibonacci(n):"])
    print("Generated output:")
    print(output[0].outputs[0].text)

    print("\n✅ Path B confirmed working end-to-end.")


if __name__ == "__main__":
    main()
