import numpy    as np
import struct
import torch
import torch.nn as nn

def expand(
	dw   : torch.Tensor,
	s_out: int,
	s_in : int,
) -> torch.Tensor:
	return (dw
		.repeat_interleave(s_out, dim=0)
		.repeat_interleave(s_in,  dim=1)
	)

class Linear(nn.Module):
	def __init__(
		self,
		s_out  : int,
		s_in   : int,
		k      : int,
		dropout: float,
		w      : torch.Tensor,
	):
		super().__init__()
		
		d_out, d_in = w.shape
		if d_in % s_in != 0 or w.dtype != torch.bfloat16:
			raise
		dw = torch.zeros(k, d_in // s_in, dtype=torch.bfloat16)
		
		self.s_out   = s_out
		self.s_in    = s_in
		self.k       = k
		self.dropout = nn.Dropout(dropout)
		self.register_parameter('w',  nn.Parameter(w,  requires_grad=False))
		self.register_parameter('dw', nn.Parameter(dw, requires_grad=True))
	
	def forward(
		self,
		x: torch.Tensor,
	) -> torch.Tensor:
		dw = self.dropout(expand(self.dw, self.s_out, self.s_in))
		y  = x @ self.w.T
		y[..., : self.k * self.s_out] += x @ dw.T
		return y

def replace(
	model         : nn.Module,
	s_out         : int,
	s_in          : int,
	k             : int,
	dropout       : float,
	target_modules: tuple[str, ...],
) -> nn.Module:
	for name, p in model.named_parameters():
		p.requires_grad = False
	
	for name, module in model.named_modules():
		if isinstance(module, nn.Linear) and any(
			name.endswith(target_module)
			for target_module in target_modules
		):
			keys   = name.split('.')
			parent = model
			for key in keys[: -1]:
				parent = parent[int(key)] if key.isdigit() else getattr(parent, key)
			
			key = keys[-1]
			old = parent[int(key)] if key.isdigit() else getattr(parent, key)
			new = Linear(s_out, s_in, k, dropout, old.weight.data)
			if key.isdigit():
				parent[int(key)] = new
			else:
				setattr(parent, key, new)
	
	return model

def to_prompt_completion(example):
	prompt = '\n'.join([
		'Below is an instruction that describes a task. Write a response that appropriately completes the request.',
		'',
		'### Instruction:',
		example['instruction'],
		'',
		'### Response:',
		'',
	])
	return {'prompt': prompt, 'completion': example['output']}

def serialize(
	model: nn.Module,
	path : str,
):
	chunks = []
	
	ws, dws = {}, {}
	for name, p in model.named_parameters():
		if name.endswith('.w'):
			ws [name] = p
		if name.endswith('.dw'):
			dws[name] = p
	
	for name, w in ws.items():
		bs = name.encode('utf-8')
		chunks.append(struct.pack('>H', len(bs)))
		chunks.append(bs)
		
		if name[: -2] + '.dw' not in dws:
			raise
		
		dw = dws[name[: -2] + '.dw']
		d_out, d_in, k = *w.shape, dw.shape[0]
		chunks.append(struct.pack('>HHH', d_out, d_in, k))
		
		f32     = dw.detach().to(torch.float32).contiguous().cpu().view(-1)
		bits    = f32.numpy().view(np.uint32)
		rounded = bits + 0x7fff + ((bits >> 16) & 1)
		bf16    = (rounded >> 16).astype(np.uint16)
		chunks.append(bf16.astype('>u2').tobytes())
	
	with open(path, 'wb') as f:
		f.write(b''.join(chunks))

def deserialize(
	path : str,
	s_out: int,
	s_in : int,
) -> dict[str, torch.Tensor]:
	dws = {}
	with open(path, 'rb') as f:
		bs = f.read()
	
	off = 0
	while off < len(bs):
		c, = struct.unpack_from('>H', bs, off); off += 2
		name = bs[off : off + c].decode('utf-8'); off += c
		
		d_out, d_in, k = struct.unpack_from('>HHH', bs, off); off += 6
		
		bf16 = np.frombuffer(bs[off : off + k * d_in // s_in * 2], dtype='>u2'); off += k * d_in // s_in * 2
		f32  = (bf16.astype(np.uint32) << 16).view(np.float32)
		
		dw = torch.zeros(d_out, d_in, dtype=torch.bfloat16)
		dw[: k * s_out] = expand(torch.from_numpy(f32).view(k, d_in // s_in), s_out, s_in)
		
		dws[name + 'eight'] = dw
	
	return dws
