import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def cifar10(args, seed=0):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	CIFAR10_train = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
											transforms.RandomHorizontalFlip(),
											transforms.RandomCrop(32, 4),
											transforms.ToTensor(),
											normalize,]), 
										download=True)
	CIFAR10_test = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
										transforms.ToTensor(),
										normalize,])
									)
	torch.manual_seed(seed)
	CIFAR10_train, CIFAR10_val = torch.utils.data.random_split(CIFAR10_train, [40000, 10000])

	train_loader = torch.utils.data.DataLoader(CIFAR10_train, batch_size=args.batch_size, shuffle=True,
												num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(CIFAR10_val, batch_size=args.batch_size, shuffle=False,
												num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(CIFAR10_test, batch_size=args.batch_size, shuffle=False,
												num_workers=4, pin_memory=True)
	return train_loader, val_loader, test_loader

def mnist(args, seed=0):
	MNIST_train = datasets.MNIST('data', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=(0.1307,), std=(0.3081,))
					   ]))
	MNIST_test = datasets.MNIST('data', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=(0.1307,), std=(0.3081,))
					   ])),
	torch.manual_seed(seed)
	MNIST_train, MNIST_val = torch.utils.data.random_split(MNIST_train, [50000, 10000])

	train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=args.batch_size, shuffle=True, 
												num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(MNIST_val, batch_size=args.batch_size, shuffle=False, 
												num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=args.batch_size, shuffle=False, 
												num_workers=4, pin_memory=True)
	return train_loader, val_loader, test_loader


DATASETS = {
	"CIFAR-10": {'loaders':cifar10, 'num_classes':10},
	"MNIST": {'loaders':mnist, 'num_classes':10}
}



