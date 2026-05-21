# Row-Restricted Reduced-Parameter Adaptation for Efficient LLM Fine-Tuning

This repository contains the PyTorch implementation of our paper.

The implementation can reproduce the main result.

You shall obey the license.

We make the ownership of the resources clarified after the paper is accepted and officially published.

# How to use

## Setup

Install package as
```shell
pip install -r requirements.txt
```

## Training

To train a model, run `train.py` as
```shell
python train.py \
  --dataset_path /path/to/dataset \
  --dropout 0.1 \
  --gradient_accumulation_steps 256 \
  --k 32 \
  --learing_rate 5e-4 \
  --max_seq_length 512 \
  --model_path /path/to/model \
  --num_train_epochs 3 \
  --output_dir /tmp \
  --packing \
  --per_device_train_batch_size 1 \
  --s 1 \
  --seed 42 \
  --target_modules q_proj k_proj v_proj up_proj down_proj \
  --warmup_steps 7
```

If you use `math_10k.json` or `commonsense_170k.json` as training datasets,
they can be available from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters).

If you use `Llama-3-8B` or `Gemma-3-12B-PT` as pre-trained models,
they can be available from [HuggingFace Models](https://huggingface.co/models).

After training, `adapter.bin` file is created in `output_dir`.

## Evaluation

To evaluate a model, `run eval.py` as
```shell
python eval.py \
  --batch_size 1 \
  --dataset AQuA \
  --model_path /path/to/model \
  --peft_path /path/to/adapter.bin \
  --s 1
```

Set `peft_path` to the path of the `adapter.bin` file generated during training.
