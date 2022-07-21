# import package
import numpy as np
import torch
import torchvision
import time
import os
from torch.utils.data import DataLoader
from colorama import Style, Fore, Back
import argparse
from tqdm import tqdm

# import file
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='models.MSBDN-RDFF.Net')
parser.add_argument('--dataset', type=str, default='utils.dataset.DatasetForInference')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--dir_path', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()


@torch.no_grad()
def evaluate(model, loader):

	print(Fore.GREEN + "==> Inference")

	start = time.time()
	model.eval()
	for image, image_name in tqdm(loader, desc='Inference'):

		if torch.cuda.is_available():
			image = image.cuda()

		pred = model(image)   
		
		file_name = os.path.join(args.save_dir, image_name[0])
		torchvision.utils.save_image(pred.cpu(), file_name)
	
	print('Costing time: {:.3f}'.format((time.time()-start)/60))
	print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


def main():

	# get the net and dataset function
	net_func = get_func(args.model)
	dataset_func = get_func(args.dataset)
	print(Back.RED + 'Using Model: {}'.format(args.model) + Style.RESET_ALL)
	print(Back.RED + 'Using Dataset: {}'.format(args.dataset) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# prepare the dataloader
	dataset = dataset_func(dir_path=args.dir_path)
	loader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=False, pin_memory=True)
	print(Style.BRIGHT + Fore.YELLOW + "# Val data: {}".format(len(dataset)) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# prepare the model
	model = net_func()


	# load the checkpoint
	assert os.path.isfile(args.checkpoint), "The checkpoint '{}' does not exist".format(args.checkpoint)
	checkpoint = torch.load(args.checkpoint)
	msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
	print(Fore.GREEN + "Loaded checkpoint from '{}'".format(args.checkpoint) + Style.RESET_ALL)
	print(Fore.GREEN + "{}".format(msg) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# move to GPU
	if torch.cuda.is_available():
		model = model.cuda()

	evaluate(model, loader)

if __name__ == '__main__':
	main()
