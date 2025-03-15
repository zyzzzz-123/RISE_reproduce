# Recursive Introspection: Teaching Language Model Agents How to Self-Improve

This is a repo for CSE 517 Final Project, reproducing RISE: Teaching Language Model Agents How to Self-Improve.
This repo is based on the original paper’s repo:  [https://github.com/cmu-mind/RISE](https://github.com/cmu-mind/RISE). 

The commands for additional experiments beyond original paper are in the [Additional Experiments](#additional-experiments) section.


## Contents
- [Install](#install)
- [Data Generation](#data-generation)
- [Finetune](#finetune)
- [Evaluate](#evaluate)
- [Citation](#citation)
- [Additional Experiments](#additional-experiments)
## Install

### Package

``` 
python3 -m venv env
source env/bin/activate
cd FastChat
pip3 install --upgrade pip
pip3 install packaging ninja wheel config tenacity gym nltk openai==0.28.1
pip3 install -e ".[model_worker,webui]"
pip3 install flash-attn==2.5.8 --no-build-isolation
pip3 install -e ".[train]"
```

### Prepare token
```
echo "OPENAI_API_KEY: '<OPENAI_API_KEY>'" > keys.cfg
huggingface-cli login
```

## Data Generation
### Collect Trajectories
#### Example1 (Distillation): Generates first iteration trajectories for GSM8K problems:
1. The first action is sampled from Llama-2-7b-chat-hf.
2. The second action (improved action) is sampled from gpt-3.5-turbo given the history.

Note: In our paper, we first performed Supervised Fine-Tuning (SFT) on Llama-2-7b-chat-hf due to its low initial performance. Once you have the fine-tuned model, replace the model path and name of Llama-2-7b-chat-hf with the path and name of your model after SFT in the following instructions.
```
# Start the controller
python -m fastchat.serve.controller &

# Start the model worker
python -m fastchat.serve.model_worker --model-path meta-llama/Llama-2-7b-chat-hf &
python -m fastchat.serve.model_worker --model-path deepseek-ai/deepseek-math-7b-instruct &

python workflow_gen.py \
    --data_path data/gsm8k/train.jsonl \
    --env gsm8k \
    --log_dir log/gsm8k/ \
    --max_turns 2 \
    --num_of_samples 1 1 \
    --models Llama-2-7b-chat-hf gpt-3.5-turbo
```

#### Example2 (Self-distillation): Generates second iteration trajectories for MATH problems:
1. The first action is sampled from Mistral-7B-Instruct-v0.2.
2. The second action (improved action) is sampled from Mistral-7B-Instruct-v0.2 using a best-of-n approach, where n = 16.

```
# Start the controller
python -m fastchat.serve.controller &

# Start the model worker
python -m fastchat.serve.model_worker --model-path mistralai/Mistral-7B-Instruct-v0.2 &

python workflow_gen.py \
    --data_path data/math/train.jsonl \
    --env MATH \
    --log_dir log/math/ \
    --max_turns 2 \
    --num_of_samples 1 16 \
    --models Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-v0.2
```

### Convert trajectories to finetune dataset
We studied four criteria in this project:
- all: Keep all actions in the trajectory
- better: Keep actions that are better than the previous action
- best: Keep only the best action in each trajectory
- success: Keep only the successful action in each trajectory

Convert the log, keeping all actions in trajectories generated in Example 1, to a finetuned dataset.
```
python workflow_convert.py \
    --filepath log/gsm8k/2_turns.json \
    --output dataset/finetune.json \
    --model_path meta-llama/Llama-2-7b-chat-hf \ ## for tokenization
    --env gsm8k \
    --criteria all \
    --remove_duplication \
    --shuffle 
```

## Finetune
```
deepspeed --master_port 20001 FastChat/fastchat/train/train_mem.py \
    --deepspeed deepspeed/zero3.json \
    --model_name_or_path <model-path> \
    --data_path <data-path> \
    --bf16 True \
    --output_dir <model-save-dir> \
    --num_train_epochs <num-of-epoch> \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --report_to wandb
```

## Evaluate 
### Sequential Evaluation
Sequentially evaluate the fine-tuned model on the GSM8K test set. Unlike [Data Generatio](#data-generation), the interaction will not terminate even when the step action is correct. To avoid test-time distribution shift, adjust the context_window to include only the most recent k interactions at the k-th iteration.

```
python workflow_eval.py \
    --model <finetuned-model-path> \
    --data_path data/gsm8k/test.jsonl \
    --env gsm8k \
    --log_dir log/ \
    --max_turns 5 \
    --num_of_samples 1 1 1 1 1 \
    --context_window 2
```


### Parallel Evaluation
Parallelly evaluate the fine-tuned model on the GSM8K test set.
```
python workflow_eval.py \
    --model <finetuned-model-path> \
    --data_path data/gsm8k/test.jsonl \
    --env gsm8k \
    --log_dir log/ \
    --max_turns 1 \
    --num_of_samples 5 \
```

### Error
If encounter Error #1 when finetune >= Llama-3.1
```
ValueError^: ^`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, got {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}^
```

Try `pip install --upgrade transformer`


## Citation
The code in this repository is mostly developed for or derived from the paper below. Please cite it if you find the repository helpful.

```
misc{qu2024recursiveintrospectionteachinglanguage,
    title={Recursive Introspection: Teaching Language Model Agents How to Self-Improve}, 
    author={Yuxiao Qu and Tianjun Zhang and Naman Garg and Aviral Kumar},
    year={2024},
    eprint={2407.18219},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2407.18219}, 
}
```

## Additional Experiments
We have done extended experiments deepseek-math-7b-instrcut to explore whether a math-specialized model benefits differently from RISE

Command to generate 1 turn collection:
```
python workflow_gen.py \
    --data_path data/gsm8k/train.jsonl \
    --env gsm8k \
    --log_dir log/gsm8k/ \
    --max_turns 1 \
    --num_of_samples 1 1 \
    --controller_address 21001 \
    --debug \
    --model_suffix ds_math_turn1 \
    --models deepseek-ai/deepseek-math-7b-instruct &
```

Command to finetune:
```
deepspeed --master_port 20007 FastChat/fastchat/train/train_mem.py \
    --deepspeed deepspeed/zero3.json \
    --model_name_or_path deepseek-ai/deepseek-math-7b-instruct \
    --data_path dataset/2_turns_deepseek_math_7b_instruct_deepseek_v3.json \
    --bf16 True \
    --output_dir /model_finetuned \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False &
```
Command to evaluate:
```
python workflow_eval.py \
    --model deepseek-ai/deepseek-math-7b-instruct \
    --data_path data/gsm8k/test.jsonl \
    --env gsm8k \
    --log_dir eval_log/dsmath_gsm8k_p \
    --max_turns 1 \
    --controller_address 21004 \
    --num_of_samples 5 &
```