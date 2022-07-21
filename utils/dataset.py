from torch.utils.data import Dataset
import torch
import glob
from torchvision import transforms
import os
from PIL import Image
from glob import glob
import json
import random


class DatasetForInference(Dataset):
	def __init__(self, dir_path):
		self.image_paths = glob(os.path.join(dir_path, '*'))
		self.transform = transforms.ToTensor()

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		input_path = self.image_paths[index]
		input_image = Image.open(input_path).convert('RGB')
		input_image = self.transform(input_image)

		_, h, w = input_image.shape
		if (h%16 != 0) or (w%16 != 0):
			input_image = transforms.Resize(((h//16)*16, (w//16)*16))(input_image)

		return input_image, os.path.basename(input_path)


class DatasetForTrain(Dataset):
	def __init__(self, meta_paths):
		self.datasets = []
		for meta_path in meta_paths:
			with open(meta_path, "r") as f:
				print("Load meta file from '{}'".format(meta_path))
				self.datasets.append(json.load(f))

		self.image_size = 224  
		self.transform = transforms.ToTensor()
		self.resize = transforms.Resize((self.image_size, self.image_size))

	def __len__(self):
		return min([len(dataset) for dataset in self.datasets])

	def __getitem__(self, index):
		dataset_label = random.randint(0, len(self.datasets)-1)
		dataset = self.datasets[dataset_label]
		target_path, input_path = dataset[random.randint(0, len(dataset)-1)]

		target_image = Image.open(target_path).convert('RGB')
		input_image = Image.open(input_path).convert('RGB')
		
		target_image = self.transform(target_image)
		input_image = self.transform(input_image)

		target_image, input_image = self.rand_crop(target_image, input_image)
		target_image, input_image = self.rand_flip(target_image, input_image)

		return target_image, input_image, dataset_label
	
	def rand_flip(self, target_image, input_image):
		if random.random() > 0.5:
			target_image = target_image.flip(2)
			input_image = input_image.flip(2)

		return target_image, input_image 
	
	def rand_crop(self, target_image, input_image):
		h, w = target_image.shape[1], target_image.shape[2]
		if h < self.image_size or w < self.image_size:
			return self.resize(input_image), self.resize(target_image)

		rr = random.randint(0, h - self.image_size)
		cc = random.randint(0, w - self.image_size)

		target_image = target_image[:, rr: rr + self.image_size, cc: cc + self.image_size]
		input_image = input_image[:, rr: rr + self.image_size, cc: cc + self.image_size]

		return target_image, input_image


class DatasetForValid(Dataset):
	def __init__(self, meta_paths):
		self.dataset = []
		for meta_path in meta_paths:
			with open(meta_path, "r") as f:
				print("Load json file from '{}'".format(meta_path))
				self.dataset.extend(json.load(f))

		self.transform = transforms.ToTensor()

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		
		target_path, input_path = self.dataset[index]
		
		target_image = Image.open(target_path).convert('RGB')
		input_image = Image.open(input_path).convert('RGB')
		
		target_image = self.transform(target_image)
		input_image = self.transform(input_image)

		_, h, w = target_image.shape
		if (h%16 != 0) or (w%16 != 0):
			target_image = transforms.Resize(((h//16)*16, (w//16)*16))(target_image)
			input_image = transforms.Resize(((h//16)*16, (w//16)*16))(input_image)

		return target_image, input_image


class Collate():
	def __init__(self, n_degrades) -> None:
		self.n_degrades = n_degrades

	def __call__(self, batch):

		target_images = [[] for _ in range(self.n_degrades)]
		input_images = [[] for _ in range(self.n_degrades)]

		for i in range(len(batch)):
			target_image, input_image, dataset_label = batch[i]
			target_images[dataset_label].append(target_image.unsqueeze(0))
			input_images[dataset_label].append(input_image.unsqueeze(0))
		
		for i in range(len(target_images)):
			if target_images[i] == []:
				return None, None
			target_images[i] = torch.cat(target_images[i])
			input_images[i] = torch.cat(input_images[i])
		target_images = torch.cat(target_images)
		
		return target_images, input_images
