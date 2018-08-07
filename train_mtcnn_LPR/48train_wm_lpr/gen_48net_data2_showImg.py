import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU

#stdsize = 48
std_size_w = 48
std_size_h = 144
stdsize = str(48)
anno_file = "label_pts.txt"
im_dir = "samples"
pos_save_dir = stdsize + "/positive"
part_save_dir = stdsize + "/part"
neg_save_dir = stdsize + '/negative'
save_dir = "./" + stdsize

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
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    #pts = map(float, annotation[5:])
    #bbox = map(float, annotation[1:5])
    bbox_tmp = map(float, annotation[1:])
    bbox = []
    bbox_pts_tmp = []
    bbox_point = []
    for i in range(0,len(bbox_tmp)):
        if i%12<4:
            bbox.append(bbox_tmp[i])
        else:
            bbox_point.append(bbox_tmp[i])
            bbox_pts_tmp.append(bbox_tmp[i])
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    boxes_point = np.array(bbox_point, dtype=np.float32).reshape(-1, 8)
    boxes_pts = []
    pts = []
    for i in range(len(bbox_pts_tmp)):
        pts.append(bbox_pts_tmp[i])
        if 0 == (1+i)%8:
            boxes_pts.append(pts)
            pts = []
    #pts = bbox_pts
    img = cv2.imread(im_path)
    cv2.namedWindow("input",1)
    cv2.imshow("input",img)
    idx += 1
    if idx % 100 == 0:
        print idx, "images done"

    height, width, channel = img.shape

    neg_num = 0
    #pts = boxes_pts[0]
    while neg_num < 100:
        #size = npr.randint(40, min(width, height) / 2)
        size_w = npr.randint(40, min(width, height) / 2)
        #size_h = npr.randint(40, min(width, height) / 2)
		#size_w = npr.randint(int(w) * 0.8, np.ceil(1.25 * w ))
        size_h = int(float(size_w)/float(3))
				
        nx = npr.randint(0, width - size_w)
        ny = npr.randint(0, height - size_h)
        crop_box = np.array([nx, ny, nx + size_w, ny + size_h])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + size_h, nx : nx + size_w, :]
        resized_im = cv2.resize(cropped_im, (std_size_h, std_size_w), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(img, (int(nx), int(ny)), (int(nx + size_w), int(ny + size_h)), (55, 55, 155), 2)
        cv2.namedWindow("input_cropped_im", 1)
        cv2.imshow("input_cropped_im", img)

        cv2.namedWindow("cropped_im", 1)
        cv2.imshow("cropped_im", cropped_im)
        cv2.waitKey(1)
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write(str(stdsize)+"/negative/%s"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    i_count = 0
    
    
    #pts = boxes_pts[i_count]
    #Test_flag = False
    #backupPts = pts[:]
    for box in boxes:
        backupPts = boxes_pts[i_count][:]
        box_point = boxes_point[i_count]
        i_count+=1
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
            pts = backupPts[:]
            #print pts
            #size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            #size_w = npr.randint(int(w * 0.8), np.ceil(1.25 * w))
            #size_h = npr.randint(int(h * 0.8), np.ceil(1.25 * h))

            # size = npr.randint(40, min(width, height) / 2)
            size_w = npr.randint(int(w * 0.8), np.ceil(1.25 * w))
            # size_h = npr.randint(40, min(width, height) / 2)
            # size_w = npr.randint(int(w) * 0.8, np.ceil(1.25 * w ))
            size_h = int(float(size_w) / float(3))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size_w / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size_h / 2, 0)
            nx2 = nx1 + size_w
            ny2 = ny1 + size_h

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size_w)
            offset_y1 = (y1 - ny1) / float(size_h)
            offset_x2 = (x2 - nx1) / float(size_w)
            offset_y2 = (y2 - ny1) / float(size_h)

            for k in range(len(pts) / 2):
                cv2.circle(img,(int(pts[k*2]), int(pts[k*2+1])), 6,(55, 155, 255), -1);

            for k in range(len(pts) / 2):
                pts[k*2] = (pts[k*2] - nx1) / float(size_w);
                pts[k*2+1] = (pts[k*2+1] - ny1) / float(size_h);

            cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
            point_min_x = min(int(box_point[0]),int(box_point[2]),\
                              int(box_point[4]),int(box_point[6]))
            point_max_x = max(int(box_point[0]),int(box_point[2]),\
                              int(box_point[4]),int(box_point[6]))
            point_min_y = min(int(box_point[1]),int(box_point[3]),\
                              int(box_point[5]),int(box_point[7]))
            point_max_y = max(int(box_point[1]),int(box_point[3]),\
                              int(box_point[5]),int(box_point[7]))





            resized_im = cv2.resize(cropped_im, (std_size_h, std_size_w), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if (point_min_x>=nx1)and(point_min_y>=ny1)and(point_max_x<=nx2)and(point_max_y<=ny2):
                cv2.rectangle(img, (int(nx1), int(ny1)), (int(nx2), int(ny2)), (0, 255, 0), 2)
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write(str(stdsize)+"/positive/%s"%p_idx + ' 1 %f %f %f %f'%(offset_x1, offset_y1, offset_x2, offset_y2))

                for k in range(len(pts)):
                    f1.write(" %f" % pts[k])
                f1.write("\n")
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                cv2.rectangle(img, (int(nx1), int(ny1)), (int(nx2), int(ny2)), (255, 0, 0), 2)
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write(str(stdsize)+"/part/%s"%d_idx + ' -1 %f %f %f %f'%(offset_x1, offset_y1, offset_x2, offset_y2))

                for k in range(len(pts)):
                    f3.write(" %f" % pts[k])
                f3.write("\n")
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

        cv2.namedWindow("input_cropped_im_pos", 1)
        cv2.imshow("input_cropped_im_pos", img)

        cv2.namedWindow("cropped_im_pos", 1)
        cv2.imshow("cropped_im_pos", cropped_im)
        cv2.waitKey(0)

        box_idx += 1
        print "%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx)

f1.close()
f2.close()
f3.close()
