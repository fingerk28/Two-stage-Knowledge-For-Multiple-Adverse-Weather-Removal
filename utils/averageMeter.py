class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def get_meter(num_meters):
	return [AverageMeter() for _ in range(num_meters)]


def update_meter(meters, values, n=1):
	for meter, value in zip(meters, values):
		meter.update(value, n=n)

	return meters