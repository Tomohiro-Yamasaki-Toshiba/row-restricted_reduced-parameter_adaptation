from datasets         import load_dataset
from torch.utils.data import DataLoader
from transformers     import AutoModelForCausalLM, AutoTokenizer

from main import to_prompt_completion, deserialize

import re
import torch

def match_aqua(instruction, label, completion):
	m = re.search(
		'Answer Choices: \\(A\\) (.*) \\(B\\) (.*) \\(C\\) (.*) \\(D\\) (.*) \\(E\\) (.*)',
		instruction,
	)
	if not m:
		raise
	
	pattern = '|'.join([
		k + '\\) ' + '\\s?'.join([
			re.escape(c)
			for c in m[i + 1].strip()
		])
		for i, k in enumerate('ABCDE')
	])
	m = re.search(pattern, completion)
	if m:
		return m.group()[0] == label
	
	m = re.search('\\W[Aa]nswer\\W.{,10}([ABCDE])(\\W|$)', completion)
	if m:
		return m.group(1)   == label
	
	return False

def match_number(instruction, label, completion):
	ms = re.findall('-?\\d+\\.?\\d*', completion)
	if len(ms) > 0:
		return abs(float(label) - float(ms[-1])) < 0.001
	
	return False

def _match_option(instruction, label, completion, options):
	ms = re.findall('|'.join(options), completion)
	if len(ms) > 0:
		return ms[0] == label
	
	return False

def match_bool(instruction, label, completion):
	return _match_option(instruction, label, completion, ['true', 'false'])

def match_solution(instruction, label, completion):
	return _match_option(instruction, label, completion, ['solution1', 'solution2'])

def match_answer(instruction, label, completion):
	return _match_option(instruction, label, completion, ['answer1', 'answer2', 'answer3', 'answer4', 'answer5'])

def match_ending(instruction, label, completion):
	return _match_option(instruction, label, completion, ['ending1', 'ending2', 'ending3', 'ending4'])

def match_option(instruction, label, completion):
	return _match_option(instruction, label, completion, ['option1', 'option2'])

def main(args):
	model     = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, attn_implementation='flash_attention_2')
	tokenizer = AutoTokenizer       .from_pretrained(args.model_path, use_fast=True)
	
	ws  = model.state_dict()
	dws = deserialize(args.peft_path, args.s, args.s)
	for k, dw in dws.items():
		ws[k] += dw
	
	model.load_state_dict(ws)
	
	tokenizer.padding_side = 'left'
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	
	model.config.pad_token_id = tokenizer.eos_token_id
	model.eval()
	
	path    = f'https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/refs/heads/main/dataset/{args.dataset}/test.json'
	dataset = load_dataset('json', data_files={'test': path})['test']
	dataset = dataset.map(
		lambda example, idx: {**to_prompt_completion(example), 'idx': idx},
		with_indices=True,
		num_proc    =1,
	)
	
	def collate_fn(batch):
		prompts = [x['prompt'] for x in batch]
		idxs    = [x['idx'   ] for x in batch]
		o = tokenizer(prompts, padding=True, return_tensors='pt')
		o['idx'] = torch.tensor(idxs, dtype=torch.long)
		return o
	
	dataloader = DataLoader(
		dataset,
		collate_fn=collate_fn,
		batch_size=args.batch_size,
		shuffle   =False,
	)
	
	pairs = []
	with torch.no_grad():
		model = model.to(0)
		for batch in dataloader:
			idxs    = batch.pop('idx').to(0)
			batch   = {k: v.to(0) for k, v in batch.items()}
			outputs = model.generate(
				**batch,
				use_cache     =True,
				max_new_tokens=256,
				temperature   =0.1,
				top_p         =0.75,
				top_k         =40,
				num_beams     =4,
			)
			texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
			for (idx, text) in zip(idxs, texts):
				_, *completions = text.split('### Response:\n')
				pairs.append((int(idx), completions[0]))
	
	instructions = dataset['instruction']
	labels       = dataset['answer'     ]
	match_fn     = {
		'AQuA'         : match_aqua,
		'gsm8k'        : match_number,
		'SVAMP'        : match_number,
		'MultiArith'   : match_number,
		'AddSub'       : match_number,
		'SingleEq'     : match_number,
		'boolq'        : match_bool,
		'piqa'         : match_solution,
		'social_i_qa'  : match_answer,
		'openbookqa'   : match_answer,
		'ARC-Easy'     : match_answer,
		'ARC-Challenge': match_answer,
		'hellaswag'    : match_ending,
		'winogrande'   : match_option,
	}[args.dataset]
	
	ok, ng = 0, 0
	for (idx, completion) in pairs:
		if match_fn(instructions[idx], labels[idx], completion):
			ok += 1
		else:
			ng += 1
	
	print(f'{ok}, {ng}')

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', required=True, type=int)
	parser.add_argument('--dataset',    required=True, choices=['AQuA', 'ARC-Challenge', 'ARC-Easy', 'AddSub', 'MultiArith', 'SVAMP', 'SingleEq', 'boolq', 'gsm8k', 'hellaswag', 'openbookqa', 'piqa', 'social_i_qa', 'winogrande'])
	parser.add_argument('--model_path', required=True)
	parser.add_argument('--peft_path',  required=True)
	parser.add_argument('--s',          required=True, type=int)
	
	args = parser.parse_args()
	
	main(args)
