# -*- coding: utf-8 -*-
"""
	descriptor: generate mtcnn training data from source image and convert it into the lmdb database
	author: Aliang 2018-01-12
"""
import numpy as np
import cv2
import lmdb
import numpy.random as npr
import data_tran_tool
import caffe
from caffe.proto import caffe_pb2
from utils import IoU

anno_file = './label.txt'

with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)

print "total num of image: %d" % num

lmdb_id = 2
dir_prefix = ''
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
item_id = 0 # 数据库的id
batch_size = 1000 #多少图片进行一次写入,防止缓存不足

# create the lmdb file
# map_size指的是数据库的最大容量，根据需求设置
if(lmdb_id == 0):
    lmdb_env_12 = lmdb.open(dir_prefix + 'mtcnn_train_12_test', map_size=5000000000)
    lmdb_txn_12 = lmdb_env_12.begin(write=True)
elif(lmdb_id == 1):
    lmdb_env_24 = lmdb.open(dir_prefix + 'mtcnn_train_24_test',   map_size=5000000000)
    lmdb_txn_24 = lmdb_env_24.begin(write=True)
else:
    lmdb_env_48 = lmdb.open(dir_prefix + 'mtcnn_train_48_test', map_size=10000000000)
    lmdb_txn_48 = lmdb_env_48.begin(write=True)


# 因为caffe中经常采用datum这种数据结构存储数据
mtcnn_datum = caffe_pb2.MTCNNDatum()

for line_idx,annotation in enumerate(annotations):

    annotation = annotation.strip().split(' ')	#每一行的数据以空白分隔符为界限
    im_path = annotation[0]					#图片的路径				
    bbox = map(float, annotation[1:])
	
    if np.size(bbox) % 4 != 0:		#标注数据有问题
		print "the annotation data in line %d is invalid, please check file %s !" % (line_idx + 1, anno_file)
		exit(-1);
    elif np.size(bbox) == 0:
		continue;

    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    boxes_num = boxes.shape[0]
    img = cv2.imread(im_path) #读取图片
    cv2.namedWindow("input",1)
    cv2.imshow("input",img)
 #   cv2.waitKey(0)


    if (line_idx+1) % 10 ==0:
		print line_idx + 1, "images done"

    height, width, channel = img.shape
    if width < 200 or height < 200:
        continue;
	
    pos_num = 0
    part_num = 0
    neg_num = 0
    num_for_each = 5
    while(pos_num < boxes_num * num_for_each and part_num < boxes_num * num_for_each and neg_num < boxes_num * num_for_each * 3):
       # print "%d images done, pos: %d part: %d neg: %d" % (line_idx + 1, p_idx, d_idx, n_idx)
        choose = npr.randint(0,100)
        
        #pos
        if(choose < 20):
            max_loop = 0
            while(1):
                max_loop += 1
                if(max_loop > boxes_num * 10):
                    break;
                box_ch = npr.randint(0,boxes_num)
                box = boxes[box_ch]
                
                x1, y1, x2, y2 = box
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                w_h = float(w)/float(h)
                print w_h

                cv2.rectangle(img, (x1, y1), (x2, y2), (55, 255, 155), 2)
                cv2.namedWindow("input", 1)
                cv2.imshow("input", img)
          #      cv2.waitKey(0)

                if max(w, h) < 10 or min(w ,h) < 5 or x1 < 0 or y1 < 0:
                    continue;
                
               # size = npr.randint(int(min(w, h)*0.8), np.ceil(1.25 * max(w, h)))
                size_w = npr.randint(int(w) * 0.8, np.ceil(1.25 * w ))
                size_h = float(size_w)/float(w_h)
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size_w
                ny2 = ny1 + size_h

                print nx1,ny1,nx2-nx1,ny2-ny1

                if nx2 > width or ny2 > height:
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size_w)
                offset_y1 = (y1 - ny1) / float(size_h)
                offset_x2 = (x2 - nx1) / float(size_w)
                offset_y2 = (y2 - ny1) / float(size_h)

                cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]


                cv2.rectangle(img, (int(nx1), int(ny1)), (int(nx2), int(ny2)), (55, 55, 155), 2)
                cv2.namedWindow("input_cropped_im", 1)
                cv2.imshow("input_cropped_im", img)

                cv2.namedWindow("cropped_im", 1)
                cv2.imshow("cropped_im", cropped_im)
                cv2.waitKey(0)

                if(lmdb_id == 0):
                    resized_im12 = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                elif(lmdb_id == 1):
                    resized_im24 = cv2.resize(cropped_im, (24, 24), interpolation=cv2.INTER_LINEAR)
                else:    
                    resized_im48 = cv2.resize(cropped_im, (48, 48), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    #save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    #cv2.imwrite(save_file, resized_im)
                    #f1.write(str(stdsize)+"/positive/%s"%p_idx + ' 1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    '''正样本的标签为 1'''
                    item_id += 1
                    
                    ''' size 12'''
                    if(lmdb_id == 0):
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im12, 1, [offset_x1, offset_y1, offset_x2, offset_y2])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_12.put(keystr, mtcnn_datum.SerializeToString())
                    
                    elif(lmdb_id == 1):
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im24, 1, [offset_x1, offset_y1, offset_x2, offset_y2])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_24.put(keystr, mtcnn_datum.SerializeToString())
                   
                    else:
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im48, 1, [offset_x1, offset_y1, offset_x2, offset_y2])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_48.put(keystr, mtcnn_datum.SerializeToString())
                    #   print("finally")   

                    # write batch
                    if(item_id) % batch_size == 0:
                        if(lmdb_id == 0):
                            lmdb_txn_12.commit()
                            lmdb_txn_12 = lmdb_env_12.begin(write=True)
                        elif(lmdb_id == 1):
                            lmdb_txn_24.commit()
                            lmdb_txn_24 = lmdb_env_24.begin(write=True)
                        elif(lmdb_id == 2):
                            lmdb_txn_48.commit()
                            lmdb_txn_48 = lmdb_env_48.begin(write=True)
                        
                        #print (item_id + 1)
                        
                    p_idx += 1
                    pos_num += 1
                    break
        #part
        elif(choose < 40):
            max_loop = 0
            while(1):
                max_loop += 1
                if(max_loop > boxes_num * 10):
                    break;
                box_ch = npr.randint(0,boxes_num)
                box = boxes[box_ch]
                
                x1, y1, x2, y2 = box
                w = x2 - x1 + 1
                h = y2 - y1 + 1

                if max(w, h) < 10 or min(w ,h) < 5 or x1 < 0 or y1 < 0:
                    continue;
                
                size = npr.randint(int(min(w, h)*0.8), np.ceil(1.25 * max(w, h)))
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
                
                if(lmdb_id == 0):
                    resized_im12 = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                elif(lmdb_id == 1):
                    resized_im24 = cv2.resize(cropped_im, (24, 24), interpolation=cv2.INTER_LINEAR)
                else:    
                    resized_im48 = cv2.resize(cropped_im, (48, 48), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.4 and IoU(crop_box, box_) < 0.65 :
                    #save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    #f3.write(str(stdsize)+"/part/%s"%d_idx + ' -1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    #cv2.imwrite(save_file, resized_im)
                    '''部分样本的标签为 -1'''
                    item_id += 1
                    
                    ''' size 12'''
                    if(lmdb_id == 0):
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im12, -1, [offset_x1, offset_y1, offset_x2, offset_y2])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_12.put(keystr, mtcnn_datum.SerializeToString())
                    
                    elif(lmdb_id == 1):
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im24, -1, [offset_x1, offset_y1, offset_x2, offset_y2])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_24.put(keystr, mtcnn_datum.SerializeToString())
                   
                    else:
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im48, -1, [offset_x1, offset_y1, offset_x2, offset_y2])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_48.put(keystr, mtcnn_datum.SerializeToString())
                    
                    # write batch
                    if(item_id) % batch_size == 0:
                        if(lmdb_id == 0):
                            lmdb_txn_12.commit()
                            lmdb_txn_12 = lmdb_env_12.begin(write=True)
                        elif(lmdb_id == 1):
                            lmdb_txn_24.commit()
                            lmdb_txn_24 = lmdb_env_24.begin(write=True)
                        elif(lmdb_id == 2):
                            lmdb_txn_48.commit()
                            lmdb_txn_48 = lmdb_env_48.begin(write=True)
                        
                    d_idx += 1
                    part_num += 1
                    break;
        #neg
        else:
            while(1):
                size = npr.randint(40, min(width, height) / 2)
                nx = npr.randint(0, width - size)
                ny = npr.randint(0, height - size)
                crop_box = np.array([nx, ny, nx + size, ny + size])

                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny : ny + size, nx : nx + size, :]

                if(lmdb_id == 0):
                    resized_im12 = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                elif(lmdb_id == 1):
                    resized_im24 = cv2.resize(cropped_im, (24, 24), interpolation=cv2.INTER_LINEAR)
                else:    
                    resized_im48 = cv2.resize(cropped_im, (48, 48), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    '''负样本的标签为 0'''
                    item_id += 1
                    
                    ''' size 12'''
                    if(lmdb_id == 0):
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im12, 0, [-1.0, -1.0, -1.0, -1.0])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_12.put(keystr, mtcnn_datum.SerializeToString())
                    
                    elif(lmdb_id == 1):
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im24, 0, [-1.0, -1.0, -1.0, -1.0])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_24.put(keystr, mtcnn_datum.SerializeToString())
                   
                    else:
                        mtcnn_datum = data_tran_tool.array_to_mtcnndatum(resized_im48, 0, [-1.0, -1.0, -1.0, -1.0])
                        keystr = '{:0>8d}'.format(item_id)
                        lmdb_txn_48.put(keystr, mtcnn_datum.SerializeToString())
                    
                    # write batch
                    if(item_id) % batch_size == 0:
                        if(lmdb_id == 0):
                            lmdb_txn_12.commit()
                            lmdb_txn_12 = lmdb_env_12.begin(write=True)
                        elif(lmdb_id == 1):
                            lmdb_txn_24.commit()
                            lmdb_txn_24 = lmdb_env_24.begin(write=True)
                        elif(lmdb_id == 2):
                            lmdb_txn_48.commit()
                            lmdb_txn_48 = lmdb_env_48.begin(write=True)
                        
                    n_idx += 1
                    neg_num += 1
                    break;
               
if (item_id+1) % batch_size != 0:
    if(lmdb_id == 0):
        lmdb_txn_12.commit()
        lmdb_env_12.close()
    elif(lmdb_id == 1):
        lmdb_txn_24.commit()
        lmdb_env_24.close()
    elif(lmdb_id == 2):
        lmdb_txn_48.commit()
        lmdb_env_48.close()
    print 'last batch'
    print "There are %d images in total" % item_id