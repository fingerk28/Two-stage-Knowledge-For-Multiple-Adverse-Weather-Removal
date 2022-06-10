# import package
from glob import glob
from sys import meta_path
import numpy as np
import torch
import torch.nn as nn
import torchvision
import time
import os
from torch.utils.data import DataLoader
from colorama import Style, Fore, Back
from tensorboardX import SummaryWriter
import yaml
import argparse
from tqdm import tqdm
import importlib
from numpy import mean
import random

# import file
from utils.dataset import *
from models.projector import *
from models.MCR import *
from utils.averageMeter import *
from utils.warmup_scheduler import *
from utils.utils import *
from utils import pytorch_ssim


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='models.MSBDN-RDFF.Net')
parser.add_argument('--dataset_train', type=str, default='utils.dataset.DatasetForTrain')
parser.add_argument('--dataset_valid', type=str, default='utils.dataset.DatasetForValid')
parser.add_argument('--meta_train', type=str, default='./meta/train/')
parser.add_argument('--meta_valid', type=str, default='./meta/valid/')
parser.add_argument('--save-dir', type=str, default=None)
parser.add_argument('--max-epoch', type=int, default=250)
parser.add_argument('--warmup-epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr-min', type=float, default=1e-6)
parser.add_argument('--batch-size', type=int, default=3)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top-k', type=int, default=3)
parser.add_argument('--train-save-image-freq', type=int, default=50)
parser.add_argument('--val-save-image-freq', type=int, default=300)
parser.add_argument('--val-freq', type=int, default=2)
parser.add_argument('--teacher', type=str, nargs='+', required=True)
parser.add_argument('--teacher_projectors', type=str, required=True)
args = parser.parse_args()

writer = SummaryWriter(os.path.join(args.save_dir, 'log'))


@torch.no_grad()
def evaluate(model, val_loader, epoch):
	print(Fore.GREEN + "==> Evaluating")
	print("==> Epoch {}/{}".format(epoch, args.max_epoch))

	psnr_list, ssim_list = [], []
	model.eval()
	start = time.time()
	pBar = tqdm(val_loader, desc='Evaluating')
	for image, target in pBar:

		if torch.cuda.is_available():
			image = image.cuda()
			target = target.cuda()

		pred = model(image)   
		
		psnr_list.append(torchPSNR(pred, target).item())
		ssim_list.append(pytorch_ssim.ssim(pred, target).item())

	print("\nResults")
	print("------------------")
	print("PSNR: {:.3f}".format(np.mean(psnr_list)))
	print("SSIM: {:.3f}".format(np.mean(psnr_list)))
	print("------------------")
	print('Costing time: {:.3f}'.format((time.time()-start)/60))
	print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	global writer
	writer.add_scalars('PSNR', {'val psnr': np.mean(psnr_list)}, epoch)
	writer.add_scalars('SSIM', {'val ssim': np.mean(ssim_list)}, epoch)

	return np.mean(psnr_list), np.mean(ssim_list)


def train_kc_stage(model, teacher_networks, s_projectors, t_projectors, train_loader, optimizer, scheduler, epoch, criterions):
	print(Fore.CYAN + "==> Training Stage 1")
	print("==> Epoch {}/{}".format(epoch, args.max_epoch))
	print("==> Learning Rate = {:.6f}".format(optimizer.param_groups[0]['lr']))
	meters = get_meter(num_meters=4)
	
	criterion_l1, criterion_scr, _ = criterions

	model.train()
	s_projectors.train()

	t_projectors.eval()
	for teacher_network in teacher_networks:
		teacher_network.eval()

	start = time.time()
	pBar = tqdm(train_loader, desc='Training')
	for target_images, input_images in pBar:
		
		# Check whether the batch contains all types of degraded data
		flag = False
		for images in input_images:
			if images == []:
				flag = True
		
		if flag: continue

		# move to GPU
		target_images = target_images.cuda()
		input_images = [images.cuda() for images in input_images]

		# Fix all teachers and collect reconstruction results and features from cooresponding teacher
		recons_from_teachers = []
		features_from_each_teachers = []
		with torch.no_grad():
			for i in range(len(teacher_networks)):
				recons, features = teacher_networks[i](input_images[i], return_feat=True)
				recons_from_teachers.append(recons)
				features_from_each_teachers.append(features)	

		recons_from_teachers = torch.cat(recons_from_teachers)
		features_from_teachers = []
		for layer in range(len(features_from_each_teachers[0])):
			features_from_teachers.append([features_from_each_teachers[i][layer] for i in range(len(teacher_networks))])

		recons_from_student, features_from_student = model(torch.cat(input_images), return_feat=True)   
		
		# Project the features to common feature space
		s_feat_list = []
		t_feat_list = []
		for i, (s_features, t_features) in enumerate(zip(features_from_student, features_from_teachers)):
			t_proj_features, _ = t_projectors[i](t_features)
			t_feat_list.append(torch.cat(t_proj_features, dim=0))

			s_proj = s_projectors[i](s_features)
			s_feat_list.append(s_proj)


		# Calculate the loss
		T_loss = criterion_l1(recons_from_student, recons_from_teachers)
		PFE_loss = 0.
		for i in range(len(s_feat_list)):
			PFE_loss += criterion_l1(s_feat_list[i], t_feat_list[i])
		SCR_loss = 0.1 * criterion_scr(recons_from_student, target_images, torch.cat(input_images))
		total_loss = T_loss + PFE_loss + SCR_loss

		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()

		meters = update_meter(meters, [total_loss.item(), T_loss.item(), PFE_loss.item(), SCR_loss.item()])
		pBar.set_postfix({'loss': '{:.3f}'.format(meters[0].avg)})

	
	print("\nResults")
	print("------------------")
	print("Total loss: {:.3f}".format(meters[0].avg))
	print("------------------")
	print('Costing time: {:.3f}'.format((time.time()-start)/60))
	print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	writer.add_scalars('loss', {'train total loss': meters[0].avg}, epoch)
	writer.add_scalars('loss', {'train T loss': meters[1].avg}, epoch)
	writer.add_scalars('loss', {'train PFE loss': meters[2].avg}, epoch)
	writer.add_scalars('loss', {'train SCR loss': meters[3].avg}, epoch)

	writer.add_scalars('lr', {'Model lr': optimizer.param_groups[0]['lr']}, epoch)
	writer.add_scalars('lr', {'Projector lr': optimizer.param_groups[1]['lr']}, epoch)

	scheduler.step()


def train_ke_stage(model, train_loader, optimizer, scheduler, epoch, criterions):
	start = time.time()
	print(Fore.CYAN + "==> Training Stage2")
	print("==> Epoch {}/{}".format(epoch, args.max_epoch))
	print("==> Learning Rate = {:.6f}".format(optimizer.param_groups[0]['lr']))
	meters = get_meter(num_meters=3)
	
	criterion_l1, _, criterion_hcr = criterions

	model.train()

	pBar = tqdm(train_loader, desc='Training')
	for target_images, input_images in pBar:

		# Check whether the batch contains all types of degraded data
		flag = False
		for images in input_images:
			if images == []:
				flag = True
		
		if flag: continue

		# move to GPU
		target_images = target_images.cuda()
		input_images = torch.cat(input_images).cuda()
		
		recons = model(input_images, return_feat=False)   
			
		G_loss = criterion_l1(recons, target_images)
		HCR_loss = 0.2 * criterion_hcr(recons, target_images, input_images)
		total_loss = G_loss + HCR_loss

		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()

		meters = update_meter(meters, [total_loss.item(), G_loss.item(), HCR_loss.item()])
		pBar.set_postfix({'loss': '{:.3f}'.format(meters[0].avg)})

	
	print("\nResults")
	print("------------------")
	print("Total loss: {:.3f}".format(meters[0].avg))
	print("------------------")
	print('Costing time: {:.3f}'.format((time.time()-start)/60))
	print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	writer.add_scalars('loss', {'train total loss': meters[0].avg}, epoch)
	writer.add_scalars('loss', {'train G loss': meters[1].avg}, epoch)
	writer.add_scalars('loss', {'train HCR loss': meters[2].avg}, epoch)

	writer.add_scalars('lr', {'Model lr': optimizer.param_groups[0]['lr']}, epoch)

	scheduler.step()


def main():

	# Set up random seed
	torch.manual_seed(19870522)
	torch.cuda.manual_seed(19870522)
	np.random.seed(19870522)
	random.seed(19870522)
	print(Back.WHITE + 'Random Seed: 19870522' + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

		
	# tensorboard
	os.makedirs(args.save_dir, exist_ok=True)
	with open(os.path.join(args.save_dir, 'args.json'), 'w') as fp:
		json.dump(vars(args), fp, indent=4)


	# get the net function
	net_func = get_func(args.model)
	dataset_train_func = get_func(args.dataset_train)
	dataset_valid_func = get_func(args.dataset_valid)
	print(Back.RED + 'Using Model: {}'.format(args.model) + Style.RESET_ALL)
	print(Back.RED + 'Using Dataset for Train: {}'.format(args.dataset_train) + Style.RESET_ALL)
	print(Back.RED + 'Using Dataset for Valid: {}'.format(args.dataset_valid) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# load teacher models
	teacher_networks = []
	for checkpoint_path in args.teacher:
		checkpoint = torch.load(checkpoint_path)
		teacher = net_func().cuda()
		teacher.load_state_dict(checkpoint['state_dict'], strict=True)
		teacher_networks.append(teacher)
		print(Fore.MAGENTA + "Loading teacher model from '{}' ...".format(checkpoint_path) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# load meta files
	meta_train_paths = glob(os.path.join(args.meta_train, '*.json'))
	meta_valid_paths = glob(os.path.join(args.meta_valid, '*.json'))


	# prepare the dataloader
	train_dataset = dataset_train_func(meta_paths=meta_train_paths)
	val_dataset = dataset_valid_func(meta_paths=meta_valid_paths)
	train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
								drop_last=True, shuffle=True, collate_fn=Collate(n_degrades=len(teacher_networks)))
	val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=False)
	print(Style.BRIGHT + Fore.YELLOW + "# Training data / # Val data:" + Style.RESET_ALL)
	print(Style.BRIGHT + Fore.YELLOW + '{} / {}'.format(len(train_dataset), len(val_dataset)) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# Prepare the projectors
	t_projectors = nn.ModuleList([])
	s_projectors = nn.ModuleList([])
	for c in [64, 128, 256, 256]:
		t_projectors.append(TeacherProjectors(c, c//2, n_teachers=len(teacher_networks)))
		s_projectors.append(StudentProjector(c, c//2))
	t_projectors = t_projectors.cuda()
	s_projectors = s_projectors.cuda()
	
	# t_projectors.load_state_dict(torch.load(args.teacher_projectors)['state_dict'], strict=True)
	print(Fore.MAGENTA + "Loading Teacher Projector Model from '{}' ...".format(args.teacher_projectors) + Style.RESET_ALL)


	# prepare the loss function
	criterions = nn.ModuleList([nn.L1Loss(), SCRLoss(), HCRLoss()]).cuda()


	# Prepare the Model
	model = net_func().cuda()


	# prepare the optimizer and scheduler
	linear_scaled_lr = args.lr * args.batch_size / 16
	optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': s_projectors.parameters()}], lr=linear_scaled_lr, betas=(0.9, 0.999), eps=1e-8)
	scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch - args.warmup_epochs, eta_min=args.lr_min)
	scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler_cosine)
	scheduler.step()


	
	# Start training pipeline
	start_epoch = 1
	top_k_state = []
	print(Fore.GREEN + "Model would be saved on '{}'".format(args.save_dir) + Style.RESET_ALL)
	for epoch in range(start_epoch, args.max_epoch + 1):
		# training
		if epoch <= 0:
			train_kc_stage(model, teacher_networks, s_projectors, t_projectors, train_loader, optimizer, scheduler, epoch, criterions)
		else:
			train_ke_stage(model, train_loader, optimizer, scheduler, epoch, criterions)

		# validating
		if epoch % args.val_freq == 0:
			psnr, ssim = evaluate(model, val_loader, epoch)
			# Check whether the model is top-k model
			top_k_state = save_top_k(model, optimizer, scheduler, top_k_state, args.top_k, epoch, args.save_dir, psnr=psnr, ssim=ssim)

		torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 's_projector': s_projectors.state_dict(), 'optimizer': optimizer.state_dict(), \
					'scheduler': scheduler.state_dict()}, os.path.join(args.save_dir, 'latest_model'))


if __name__ == '__main__':
	main()
