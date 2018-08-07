import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU

stdsize = 12
anno_file = "label.txt"
im_dir = "samples"
pos_save_dir = str(stdsize) + "/positive"
part_save_dir = str(stdsize) + "/part"
neg_save_dir = str(stdsize) + '/negative'
save_dir = "./" + str(stdsize)

def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

mkr(save_dir)
mkr(pos_save_dir)
mkr(part_save_dir)
mkr(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_' + str(stdsize) + '.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_' + str(stdsize) + '.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_' + str(stdsize) + '.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print "%d pics in total" % num
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

for annotation in annotations:
	idx += 1
	if idx >= 0:
		annotation = annotation.strip().split(' ')
		im_path = annotation[0]
		bbox = map(float, annotation[1:]) 
		boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
		img = cv2.imread(im_path)
		
		if idx % 100 == 0:
			print idx, "images done"

		height, width, channel = img.shape
		if width > 120 and height > 120:
			neg_num = 0
			while neg_num < 100:
				size = npr.randint(40, min(width, height) / 2)
				nx = npr.randint(0, width - size)
				ny = npr.randint(0, height - size)
				crop_box = np.array([nx, ny, nx + size, ny + size])

				Iou = IoU(crop_box, boxes)

				cropped_im = img[ny : ny + size, nx : nx + size, :]
				resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

				if np.max(Iou) < 0.3:
					# Iou with all gts must below 0.3
					save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
					f2.write(str(stdsize)+"/negative/%s"%n_idx + ' 0\n')
					cv2.imwrite(save_file, resized_im)
					n_idx += 1
					neg_num += 1
			for box in boxes:
				# box (x_left, y_top, x_right, y_bottom)
				x1, y1, x2, y2 = box
				w = x2 - x1 + 1
				h = y2 - y1 + 1

				# ignore small faces
				# in case the ground truth boxes of small faces are not accurate
				if max(w, h) < 12 or x1 < 0 or y1 < 0:
					continue

				# generate positive examples and part faces
				for i in range(50):
					size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

					# delta here is the offset of box center
					delta_x = npr.randint(-w * 0.2, w * 0.2)
					delta_y = npr.randint(-h * 0.2, h * 0.2)

					nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
					ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
					nx2 = nx1 + size
					ny2 = ny1 + size

					if nx2 > width or ny2 > height:
						continue
					crop_box = np.array([nx1, ny1, nx2, ny2])

					offset_x1 = (x1 - nx1) / float(size)
					offset_y1 = (y1 - ny1) / float(size)
					offset_x2 = (x2 - nx1) / float(size)
					offset_y2 = (y2 - ny1) / float(size)

					cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
					resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

					box_ = box.reshape(1, -1)
					if IoU(crop_box, box_) >= 0.65:
						save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
						f1.write(str(stdsize)+"/positive/%s"%p_idx + ' 1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
						cv2.imwrite(save_file, resized_im)
						p_idx += 1
					elif IoU(crop_box, box_) >= 0.4:
						save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
						f3.write(str(stdsize)+"/part/%s"%d_idx + ' -1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
						cv2.imwrite(save_file, resized_im)
						d_idx += 1

				box_idx += 1
			#	print "%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx)

f1.close()
f2.close()
f3.close()
