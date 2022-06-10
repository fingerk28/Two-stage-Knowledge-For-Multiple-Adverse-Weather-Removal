import torch
import os
from colorama import Style, Fore, Back
import numpy as np
import importlib


def get_func(path):
	module = path[:path.rfind('.')]
	model_name = path[path.rfind('.') + 1:]
	mod = importlib.import_module(module)
	net_func = getattr(mod, model_name)

	return net_func


def save_top_k(model, optimizer, scheduler, top_k_state, k, epoch, save_dir, psnr, ssim):
	flag = False
	popped_state = {}
	model_path = os.path.join(save_dir, 'epoch_{}_psnr{:.3f}_ssim{:.3f}'.format(epoch, psnr, ssim))

	if len(top_k_state) < k or psnr >= top_k_state[-1]['psnr']:
		
		if len(top_k_state) >= k:
			popped_state = top_k_state.pop()
			os.remove(os.path.join(save_dir, 'epoch_{}_psnr{:.3f}_ssim{:.3f}'.format(popped_state['epoch'], popped_state['psnr'], popped_state['ssim'])))

		flag = True
		top_k_state.append({'epoch': epoch, 'psnr': psnr, 'ssim': ssim})
		scheduler = scheduler.state_dict() if scheduler is not None else None
		torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}, model_path)

	top_k_state.sort(key = lambda s: s['psnr'], reverse = True)
	if flag:
		if popped_state == {}:
			print(Back.RED + 'PSNR: {:.3f} , length of buffer < {}'.format(psnr, k))
		else:
			print(Back.RED + 'PSNR: {:.3f}  >=  last PSNR: {:.3f}'.format(psnr, popped_state['psnr']))
		print('Save the better model!!!' + Style.RESET_ALL)
		print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)
	else:
		print(Back.GREEN + 'PSNR: {:.3f}  <  rank-3 PSNR: {:.3f}'.format(psnr, top_k_state[-1]['psnr']))
		print('Do not save this model, QQQ' + Style.RESET_ALL)
		print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	return top_k_state


@torch.no_grad()
def torchPSNR(tar_img, prd_img):
	imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
	rmse = (imdff**2).mean().sqrt()
	ps = 20 * torch.log10(1/rmse)
	return ps


def cpt_psnr(img, img_gt, PIXEL_MAX):
	assert img.max() <= 1. and img.min() >= 0.
	assert img_gt.max() <= 1. and img_gt.min() >= 0.
	mse = np.mean((img - img_gt) ** 2)
	psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
	
	return psnr


def rgb2ycbcr(img, only_y=True):
	'''same as matlab rgb2ycbcr
	only_y: only return Y channel
	Input:
		uint8, [0, 255]
		float, [0, 1]
	'''
	if isinstance(img, torch.Tensor):
		img = img.clamp(0., 1.)
		img = img.cpu().detach().permute(1, 2, 0).numpy()

	in_img_type = img.dtype
	img.astype(np.float32)
	if in_img_type != np.uint8:
		img *= 255.
	
	# convert
	if only_y:
		rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
	else:
		rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
							  [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
	if in_img_type == np.uint8:
		rlt = rlt.round()
	else:
		rlt /= 255.
	if rlt.ndim != 3:
		rlt = np.expand_dims(rlt, axis=-1)
		
	return rlt.astype(in_img_type)

