for loading the checkpint u would need a diff studio- there use merge_lora_experts..py,export_vllm_checkpoint.py_push_checkpoint.py,(training_config.py,configuration_lora_moe.py,peft_experts.py,modelling.py,requirements.txt(same as malora original repo))
on loading run merge file, gives diff but its bf16 rounding noise(this might be also due to some other issue but as of now that is the assued cause), then run export,then use push_checkpoint to push
also u can entirely skkip checkpoint (i have alrdy uploaded on ur hf for 10k_1ep_vllm in consideration with maloraa_10k_1ep_aton checkpoint ) so u can directly go to vllm and pull out
for vllm use diff studio first run requirements.txt (also once read that file, commented imp info there) and then run test_vllm.py, on that studio, have export checkpoint foleder(read pull_checkpoint for more info), and lora_moe_vllm,test_vllm
also before test_vllm.py use this for loading tokenniser(also maybe check if its the same as abse model,maybe thats the issue)python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-3B-Instruct', trust_remote_code=True)
tok.save_pretrained('vllm_export_checkpoint')
"
