import torch
import numpy as np
from tqdm import tqdm

RETRAINING_OPTIONS = ["none", "fine-tuning", "weight-rewinding", "lr-rewinding"]
SPARSITY_REG_OPTIONS= ["none", "l1", "admm"]

def compute_accuracy(model, data_loader, device):
	with torch.no_grad():
		model.eval()
		acc = 0
		for batch in tqdm(data_loader):
			x,y = batch
			x = x.to(device)
			y = y.to(device)
			output = model((x,y), get_prediction=True)
			pred = torch.argmax(output, dim=1)
			acc += np.sum(pred.eq(y).cpu().data.numpy())
	return acc/len(data_loader.dataset)