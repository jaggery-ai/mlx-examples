convert:
	python lora/convert.py --hf-path ../../Meta-Llama-3-8B-Instruct --mlx-path ./mlx_model -q

train-lora:
	python lora/lora.py --model ./mlx_model/llama3 --train --data ./data --iters 1000 --steps-per-eval 200

train-lora-resume:
	python lora/lora.py --model ./mlx_model/llama3 --train --data ./data --iters 1000 --steps-per-eval 200 --resume-adapter-file $(ADAPTER_FILE)

talk:
	python lora/lora.py --model ./mlx_model/llama3 --adapter-file $(ADAPTER_FILE) --max-tokens 2048 --converse