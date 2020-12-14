try:
	import torch, tqdm, h5py
except ModuleNotFoundError:
	print(
			"Please install the required packages."
			"torch_weight_pruning requires torch, tqdm and h5py."
		)