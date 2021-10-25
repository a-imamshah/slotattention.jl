from multi_object_datasets import clevr_with_masks
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

import matplotlib.image as img
import numpy as np
from PIL import Image

tf_records_path = '/kuacc/users/ashah20/datasets/clevr_with_masks/clevr_with_masks_train.tfrecords'
batch_size = 32

dataset = clevr_with_masks.dataset(tf_records_path)
batched_dataset = dataset.batch(batch_size)
iterator = batched_dataset.make_one_shot_iterator()
data = iterator.get_next()


i = 0
counter = 1
val_images = 320

max_n_objects = 6

for element in dataset:
    i+=1
    
    if(i<=70000):
        
        mask_array = element["mask"]
        mask_array = np.array(mask_array)
        
        if(np.sum(np.sum(mask_array[max_n_objects+1,:,:,0])) == 0):
        
        
            a_img = element["image"]
            a_img = np.array(a_img)
            x='{}'.format(i).zfill(5)
            x = '/kuacc/users/ashah20/datasets/clevr_with_masks/clevr6/train/images/' + x + '.jpg'
            img.imsave(x, a_img)

            for j in range(0,max_n_objects+1):
                a_mask = mask_array[j,:,:,0]
                msk = Image.fromarray(a_mask)

                y='{}'.format(i).zfill(5)
                y = '/kuacc/users/ashah20/datasets/clevr_with_masks/clevr6/train/masks/' + y + '_' + '{}.jpg'.format(j).zfill(6)
                msk.save(y)
            
            
    elif(counter <= val_images):
        mask_array = element["mask"]
        mask_array = np.array(mask_array)
        
        if(np.sum(np.sum(mask_array[max_n_objects+1,:,:,0])) == 0):
            counter+=1
        
            a_img = element["image"]
            a_img = np.array(a_img)
            x='{}'.format(i).zfill(5)
            x = '/kuacc/users/ashah20/datasets/clevr_with_masks/clevr6/val/images/' + x + '.jpg'
            img.imsave(x, a_img)

            for j in range(0,max_n_objects+1):
                a_mask = mask_array[j,:,:,0]
                msk = Image.fromarray(a_mask)

                y='{}'.format(i).zfill(5)
                y = '/kuacc/users/ashah20/datasets/clevr_with_masks/clevr6/val/masks/' + y + '_' + '{}.jpg'.format(j).zfill(6)
                msk.save(y)
        
    elif (counter > val_images):
        break
