from datasets     import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl          import SFTConfig, SFTTrainer

from main import replace, to_prompt_completion, serialize

import torch

def main(args):
	torch.cuda.set_device(0)
	set_seed(args.seed)
	
	model     = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, attn_implementation='flash_attention_2')
	tokenizer = AutoTokenizer       .from_pretrained(args.model_path, use_fast=True)
	
	model.gradient_checkpointing_enable()
	model.enable_input_require_grads()
	
	tokenizer.padding_side = 'right'
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	
	model = replace(model, args.s, args.s, args.k, args.dropout, args.target_modules)
	
	dataset = load_dataset('json', data_files=args.dataset_path, split='train')
	dataset = dataset.map(
		to_prompt_completion,
		num_proc      =1,
		remove_columns=dataset.column_names,
	).shuffle(seed=args.seed)
	
	sft_config = SFTConfig(
		bf16                       =True,
		fp16                       =False,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		gradient_checkpointing     =True,
		learning_rate              =args.learning_rate,
		max_length                 =args.max_seq_length,
		num_train_epochs           =args.num_train_epochs,
		output_dir                 =args.output_dir,
		packing                    =args.packing,
		per_device_train_batch_size=args.per_device_train_batch_size,
		report_to                  ='none',
		save_total_limit           =1,
		seed                       =args.seed,
		warmup_steps               =args.warmup_steps,
	)
	
	trainer = SFTTrainer(
		args            =sft_config,
		model           =model,
		processing_class=tokenizer,
		train_dataset   =dataset,
	)
	
	trainer.train()
	serialize(model, f'{args.output_dir}/adapter.bin')

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_path',                required=True)
	parser.add_argument('--dropout',                     required=True, type=float)
	parser.add_argument('--gradient_accumulation_steps', required=True, type=int)
	parser.add_argument('--k',                           required=True, type=int)
	parser.add_argument('--learning_rate',               required=True, type=float)
	parser.add_argument('--max_seq_length',              required=True, type=int)
	parser.add_argument('--model_path',                  required=True)
	parser.add_argument('--num_train_epochs',            required=True, type=float)
	parser.add_argument('--output_dir',                  required=True)
	parser.add_argument('--packing',                     action='store_true')
	parser.add_argument('--per_device_train_batch_size', required=True, type=int)
	parser.add_argument('--s',                           required=True, type=int)
	parser.add_argument('--seed',                        required=True, type=int)
	parser.add_argument('--target_modules',              nargs='+')
	parser.add_argument('--warmup_steps',                required=True, type=int)
	args = parser.parse_args()
	
	main(args)
