import os
import torch
import torch.nn.utils.prune as prune
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from livelossplot import PlotLosses

from utils import SPARSITY_REG_OPTIONS, RETRAINING_OPTIONS

class OneShotPruning:
	def __init__(self, model, parameters_to_prune, optimizer, lr_scheduler, train_dataloader, val_dataloader, compute_val_performance, device, args):
		self.model = model
		self.parameters_to_prune = parameters_to_prune
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler

		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader
		self.compute_val_performance = compute_val_performance
		self.device = device
		
		self.args = args
		self.check_args()
		self.rewind_ckpt_path = args.model_path + '_rewind_late_reset'

	def check_args(self,):
		if self.args.seed is not None:
			torch.manual_seed(self.args.seed)
			np.random.seed(self.args.seed)
			cudnn.benchmark = True
		assert self.args.sparsity_reg in SPARSITY_REG_OPTIONS, "sparsity_reg should be one of {}".format(SPARSITY_REG_OPTIONS)
		assert self.args.retrain_mode in RETRAINING_OPTIONS, "retrain_mode should be one of {}".format(RETRAINING_OPTIONS)

	def _l1_reg(self):
		res = 0.
		for name, param in self.model.named_parameters():
			res += param.norm(1)
		return self.args.l1_penalty * res

	def _train(self, ckpt=None, is_retrain=False, plot_verbosity=True):
		print("Note that the sparsity regularizations are not implemented yet...")
		if (ckpt):
			"""in case training needs to be started from a checkpoint (Eg.: Case training a pre-trained model)"""
			self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
			if self.lr_scheduler is not None:
				self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
			self.model.load_state_dict(ckpt['model_state_dict'])

		args = self.args
		n_epochs = args.retrain_epochs if is_retrain else args.train_epochs
		best_ep, best_loss = 0, np.inf
		best_ckpt_path = args.model_path+'_best' + ("_retrain" if is_retrain else "")
		liveloss = PlotLosses()
		loss_history = []

		if (not is_retrain):
			"""Get the rewind epoch details for checkpointing."""
			nB = len(self.train_dataloader.dataset)/args.batch_size
			rewind_epochs = args.rewind_epoch
			rewind_ep = int(rewind_epochs)
			rewind_residual_batch = nB * (rewind_epochs - rewind_ep)

		for ep in range(n_epochs):
			# TRAINING
			epoch_train_loss = 0.
			self.model.train()
			for i, batch in tqdm(enumerate(self.train_dataloader)):
				if (not is_retrain) and (ep==rewind_ep and i>=rewind_residual_batch):
					# if (args.retrain_mode=='weight-rewinding') 
					# Checkpoint optimizer, lr_scheduler, and weights after 1.4 epochs for weight/ lr rewinding purposes
					w_rewind_ckpt = {
									"model_state_dict": self.model.state_dict(),
									"optimizer_state_dict": self.optimizer.state_dict(),
									"lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
									"epoch": rewind_epochs
								}
					torch.save(w_rewind_ckpt, self.rewind_ckpt_path)
				# perform the training
				loss = self.model(batch) # model performs the outpur computation and the loss computation
				self.optimizer.zero_grad()
				if self.args.sparsity_reg=='l1':
					loss += self._l1_reg()
				loss.backward()
				self.optimizer.step()
				# store the loss for logging
				epoch_train_loss += loss.cpu().data.item()*len(batch[0])
				# step learning rate
				if self.lr_scheduler is not None:
					self.lr_scheduler.step()
			epoch_train_loss /= len(self.train_dataloader.dataset)
			# VALIDATION
			with torch.no_grad():
				epoch_val_loss = 0.
				self.model.eval()
				for batch in tqdm(self.val_dataloader):
					loss = self.model(batch)
					epoch_val_loss += loss.cpu().data.item()*len(batch[0])
				epoch_val_loss /= len(self.val_dataloader.dataset)
			# PLOT THE METRICS
			if plot_verbosity:
				plot_dict = {"loss": epoch_train_loss, "val_loss": epoch_val_loss}
				if self.compute_val_performance is not None:
					plot_dict.update({"val_performance": self.compute_val_performance(self.model, self.val_dataloader, self.device)})
				liveloss.update(plot_dict)
				liveloss.send()
			loss_history.append( (epoch_train_loss, epoch_val_loss) )
			# DO THE EARLY STOPPING
			if(args.use_early_stop):
				if(epoch_train_loss > best_loss):
					if(args.patience + best_ep < ep):
						break
				else:
					best_ep = ep
					best_loss = epoch_train_loss
					best_ckpt = {
									"model_state_dict": self.model.state_dict(),
									"optimizer_state_dict": self. optimizer.state_dict(),
									"lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
									"epoch": best_ep
								}
					torch.save(best_ckpt, best_ckpt_path)
		if not (ep==best_ep):
			best_ckpt = torch.load(best_ckpt_path)
			self.model.load_state_dict(best_ckpt['model_state_dict'])
			self.lr_scheduler.load_state_dict(best_ckpt['lr_scheduler_state_dict'])
			self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
		return loss_history

	def _prune(self, ):
		prune.global_unstructured(self.parameters_to_prune(self.model), pruning_method=prune.L1Unstructured, amount=(1-self.args.prune_ratio))
		# remove the pruning reparameterization
		for module, name in self.parameters_to_prune(self.model):
			prune.remove(module, 'weight')

	def _prepare_model(self,):
		"""
		Prepares the mask according to the current zeroed-weights, and 
		registers a backward hook to prevent the parameters from being updated.
		To be called after self._prune().

		Returns: a list of tensor hook handles.
				to remove the hooks, call <handle>.remove().
		"""
		self.model.train()
		backward_hook_handles = []
		for module, name in self.parameters_to_prune(self.model):
			mask = (1-(module.weight.data == 0.)*1.)
			def hook(grad, mask=mask):
				return grad.data.mul(mask)
			# reggister a backward hook for the model and append the handles in a list
			backward_hook_handles.append( module.weight.register_hook(hook) )
		return backward_hook_handles

	def _retrain(self, rewind_ckpt_path=None):
		if(rewind_ckpt_path is None):
			rewind_ckpt_path = self.rewind_ckpt_path
		# at this point self.model should be pruned
		backward_hook_handles = self._prepare_model()
		ckpt = torch.load(rewind_ckpt_path)
		if(self.args.retrain_mode == 'lr-rewinding'):
			if (self.lr_scheduler is not None):
				self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])

		elif(self.args.retrain_mode == 'weight-rewinding'):
			self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
			if (self.lr_scheduler is not None):
				self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
			# save the masks from the pruned model
			masks = []
			for module, name in self.parameters_to_prune(self.model):
				masks.append(1-(module.weight.data == 0.)*1.)
			# load the model weights from the rewind checkpoints
			self.model.load_state_dict(ckpt['model_state_dict'])
			# zero out the pruned weights as per the mask
			for (module, name), mask in zip(self.parameters_to_prune(self.model), masks):
				module.weight.data.mul_(mask)

		elif(self.args.retrain_mode == 'fine-tuning'):
			pass
		self._train(is_retrain=True)
		return backward_hook_handles

	def get_ticket(self):
		args = self.args
		print("TRAINING PHASE:")
		loss_history1 = self._train()
		best_ckpt1 = args.model_path + '_best' 
		val_acc1 = self.compute_val_performance(self.model, self.val_dataloader, self.device)
		print("PRUNING PHASE:")
		self._prune()
		val_acc2 = self.compute_val_performance(self.model, self.val_dataloader, self.device)
		print("RE-TRAINING PHASE:")
		loss_history3 = self._retrain()
		best_ckpt3 = args.model_path + '_best_retrain'
		val_acc3 = self.compute_val_performance(self.model, self.val_dataloader, self.device)
		if(val_acc2>=val_acc1):
			print("Congrats! this is a winning ticket. \n val accuracy (original): {}\n val accuracy (pruned): {} \n val accuracy (retrained): {} ".format(val_acc1, val_acc2, val_acc3))
		else:
			print(" val accuracy (original): {}\n val accuracy (pruned): {} \n val accuracy (retrained): {} ".format(val_acc1, val_acc2, val_acc3))
		return self.model, best_ckpt1, best_ckpt3, loss_history1, loss_history3
