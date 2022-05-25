import os
import sys
import os
import sys
from turtle import shape
from prettytable import PrettyTable
from pie_intent import PIEIntent
from pie_predict import PIEPredict

from pie_data import PIE
import pickle as pkl
import random
from sklearn.utils import shuffle
import keras
import numpy as np
from PIL import Image
import pickle as pkl
import shutil
import tensorflow as tf
# import tensorflow.keras as keras
from keras.preprocessing.image import img_to_array,array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def train_test_split_data(shuffled=True):
    # Used this line as our filename array is not a numpy array.
    if shuffled:
        filenames= np.load('FROM_dataset/filenames_shuffled.npy')
        labels= np.load('FROM_dataset/labels_shuffled.npy')
    else:
        filenames= np.load('FROM_dataset/filenames.npy')
        labels= np.load('FROM_dataset/labels.npy')

    filenames= np.array(filenames)
    labels= np.array(labels)
    train_filenames, val_filenames,labels_train, labels_val = train_test_split(
        filenames, labels, test_size=0.2, random_state=1)

    print('train_filenames.shape: ',train_filenames.shape) # (3800,)
    print('labels_train.shape: ',labels_train.shape)           # (3800, 12)

    print('val_filenames.shape: ',val_filenames.shape)   # (950,)
    print('labels_val.shape: ',labels_val.shape)             # (950, 12)

    # You can save these files as well. As you will be using them later for training and validation of your model.
    np.save('FROM_dataset/train_filenames.npy', train_filenames)
    np.save('FROM_dataset/labels_train.npy', labels_train)

    np.save('FROM_dataset/val_filenames.npy', val_filenames)
    np.save('FROM_dataset/labels_val.npy', labels_val)
def bbox_sanity_check(img, bbox):
    '''
    This is to confirm that the bounding boxes are within image boundaries.
    If this is not the case, modifications is applied.
    This is to deal with inconsistencies in the annotation tools
    '''
    img_width, img_heigth = img.size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_heigth:
        bbox[3] = img_heigth - 1
    return bbox

def img_pad(img, mode = 'warp', size = 224):
    '''
    Pads a given image.
    Crops and/or pads a image given the boundries of the box needed
    img: the image to be coropped and/or padded
    bbox: the bounding box dimensions for cropping
    size: the desired size of output
    mode: the type of padding or resizing. The modes are,
        warp: crops the bounding box and resize to the output size
        same: only crops the image
        pad_same: maintains the original size of the cropped box  and pads with zeros
        pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
        the desired output size in that direction while maintaining the aspect ratio. The rest of the image is
        padded with zeros
        pad_fit: maintains the original size of the cropped box unless the image is biger than the size in which case
        it scales the image down, and then pads it
    '''
    assert(mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = img.copy()
    if mode == 'warp':
        warped_image = image.resize((size,size),Image.NEAREST)
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same','pad_resize','pad_fit']:
        img_size = image.size  # size is in (width, height)
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or  \
            (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0]*ratio),int(img_size[1]*ratio)])
            image = image.resize(img_size, Image.NEAREST)
        padded_image = Image.new("RGB", (size, size))
        padded_image.paste(image, ((size-img_size [0])//2,
                    (size-img_size [1])//2))
        return padded_image

def squarify(bbox, squarify_ratio, img_width):
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    # width_change = float(bbox[4])*self._squarify_ratio - float(bbox[3])
    bbox[0] = bbox[0] - width_change/2
    bbox[2] = bbox[2] + width_change/2
    # bbox[1] = str(float(bbox[1]) - width_change/2)
    # bbox[3] = str(float(bbox[3]) + width_change)
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0
    
    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
        bbox[0] = bbox[0]-bbox[2] + img_width
        bbox[2] = img_width
    return bbox

def flatten(seq):
    temp_list=[] # list to hold flattened value 
    for i in seq:
        flattened =  [item for sublist in seq[i] for item in sublist]
        temp_list.append(flattened)
        # print('--------',i,' added as the ',len(temp_list),' element')
        # print("size of ",i,' is ',len(flattened)) 

        
    return   {'image': temp_list[0],
                'bbox': temp_list[1],
                'occlusion': temp_list[2],
                'intention_prob': temp_list[3],
                'intention_binary': temp_list[4],
                'ped_id': temp_list[5]} 

def jitter_bbox(img_path,bbox, mode, ratio):
    '''
    This method jitters the position or dimentions of the bounding box.
    mode: 'same' returns the bounding box unchanged
          'enlarge' increases the size of bounding box based on the given ratio.
          'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
          'move' moves the center of the bounding box in each direction based on the given ratio
          'random_move' moves the center of the bounding box in each direction by randomly sampling a value in [-ratio,ratio)
    ratio: The ratio of change relative to the size of the bounding box. For modes 'enlarge' and 'random_enlarge'
           the absolute value is considered.
    Note: Tha ratio of change in pixels is calculated according to the smaller dimension of the bounding box.
    '''
    assert(mode in ['same','enlarge','move','random_enlarge','random_move']), \
            'mode %s is invalid.' % mode

    if mode == 'same':
        return bbox

    img = load_img(img_path)
    img_width, img_heigth = img.size

    if mode in ['random_enlarge', 'enlarge']:
        jitter_ratio  = abs(ratio)
    else:
        jitter_ratio  = ratio

    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample()*jitter_ratio
    elif mode == 'random_move':
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
        jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

    jit_boxes = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]

        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change

        if mode in ['enlarge','random_enlarge']:
            b[0] = b[0] - width_change //2
            b[1] = b[1] - height_change //2
        else:
            b[0] = b[0] + width_change //2
            b[1] = b[1] + height_change //2

        b[2] = b[2] + width_change //2
        b[3] = b[3] + height_change //2

        # Checks to make sure the bbox is not exiting the image boundaries
        b =  bbox_sanity_check(img, b)
        jit_boxes.append(b)
    # elif crop_opts['mode'] == 'border_only':
    return jit_boxes


class My_Custom_Generator(tf.keras.utils.Sequence):#keras.utils.all_utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    fnames=[]
    labels=[]
    for file_name, label_name in zip (batch_x,batch_y):
        with open(file_name,'rb') as f:
            fnames.append(np.array(pkl.load(f)))
            labels.append(np.array(label_name))


    return (np.array(fnames)/255.0), np.array(labels)
               
def image_pre_process(data,replace=False,save_shuffled=True):
    image_list=[]
    i = 0 
    name_array=[]
    label_array=[]
    if not os.listdir('From_dataset/Pre_Processed_images_all') or replace:
        if replace:
            print('Replacing Previous Data...')
        else:
            print('No Previous Data Found')
        print('Image Pre_Processing started ...')
        img_save_folder = os.path.join('From_dataset/Pre_Processed_images_all')
        for imp, bb, p,intention in zip(data['image'],data['bbox'], data['ped_id'],data['intention_binary']):
            update_progress(i / len(data['image']))
            # SAVE PICKLED image
            set_id = imp.split('/')[-3]
            print('set_id : ', set_id)
            vid_id = imp.split('/')[-2]
            print('vid_id : ', vid_id)
            img_name = imp.split('/')[-1].split('.')[0]
            print('img_name : ', img_name)
            img_save_folder = os.path.join('From_dataset/Pre_Processed_images_all')
            img_save_path = os.path.join(img_save_folder, img_name+'_'+p[0]+'.pkl')
            if not os.path.exists(img_save_folder):
                    os.makedirs(img_save_folder)
            img_data = load_img(imp)
            bbox = jitter_bbox(imp, [bb],'enlarge', 2)[0]
            bbox = squarify(bbox, 1, img_data.size[0])
            bbox = list(map(int,bbox[0:4]))
            cropped_image = img_data.crop(bbox)
            img_data = img_pad(cropped_image, mode='pad_resize', size=224) 
            image_array = img_to_array(img_data)
            with open(img_save_path, 'wb') as fid:
                pkl.dump(image_array, fid, pkl.HIGHEST_PROTOCOL)
            i+=1
            name_array.append(img_save_path)
            label_array.append(intention[0])
            print('Number of processed info Labels: ', len(label_array),' Fnames: ',len(name_array),' Images: ',len(data['image']))
        np.save('From_dataset/filenames.npy',name_array)
        np.save('From_dataset/labels.npy',label_array)
        if(save_shuffled):
            shuffling(name_array,label_array)
            print('shuffled filenames and labels saved!')
    else:
        print('-------------------Previously Processed Image Data Found skipping processing!-------------------')
    # Save labels as pickles and save all images in one folder
    # save_labels_images(data,replace_data=False,shuffled=save_shuffled)

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(barLength*progress))
    text = "\r[{}] {:0.2f}% {}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def save_labels_images(data,replace_data=False,shuffled=False):
    counter = 0
    name_array=[]
    label_array=[]
    if not os.listdir('FROM_dataset/Pre_processed_all_images') or replace_data: # check if this folder has images
        if replace_data:
            print('Replacing Previous Data...')
        else:
            print('No Previous Data Found')
        
        print('Putting all images in one Folder, creating name and label files ...')
        for imp, bb, p,intention in zip(data['image'],data['bbox'], data['ped_id'],data['intention_binary']):
            # if counter == 100:
            #     break
            update_progress(counter / len(data['image']))
            
            set_id = imp.split('/')[-3]
            # print('set_id : ', set_id)
            vid_id = imp.split('/')[-2]
            # print('vid_id : ', vid_id)
            img_name = imp.split('/')[-1].split('.')[0]
            # print('img_name : ', img_name)
            img_save_folder = os.path.join('FROM_dataset/Pre_Processed_images', set_id, vid_id)
            fullpath = os.path.join(img_save_folder, img_name+'_'+p[0]+'.pkl')
            # print('fullpath : ', fullpath)
            shutil.copy(fullpath,'FROM_dataset/Pre_processed_all_images')
            name_array.append(fullpath)
            label_array.append(intention[0])
            #tf.keras.preprocessing.image.array_to_img(image).show()
            counter+=1
        #label_catagory = to_categorical(label_array)
        print('Number of processed info Labels: ', len(label_array),' Fnames: ',len(name_array),' Images: ',len(data['image']))
        np.save('FROM_dataset/filenames.npy',name_array)
        np.save('FROM_dataset/labels.npy',label_array)
        if(shuffled):
            shuffling(name_array,label_array)
            print('shuffled filenames and labels saved!')
    else:
        print('-------------------Previously Processed Image Data Found skipping Image Migration!-------------------')

def shuffling(filenames,y_labels_one_hot):
    filenames_shuffled, y_labels_one_hot_shuffled = shuffle(filenames, y_labels_one_hot)
    # saving the shuffled file.
    # you can load them later using np.load().
    print(' ----------------------- Everday am shufling ----------------------------')
    np.save('From_dataset/test_labels_shuffled.npy', y_labels_one_hot_shuffled)
    np.save('From_dataset/test_filenames_shuffled.npy', filenames_shuffled)

def data_size(beh_seq_train,print_details=True):
        all_img = 0
        for i in range(len(beh_seq_train['image'])):
            last = len(beh_seq_train['image'][i]) - 1 
            all_img+=last+1
            for j in range(len(beh_seq_train['image'][i])):
                if print_details:
                    print('--------Sequence-',i,' Image: ',j,'------------------------------')
                    print("Image",beh_seq_train['image'][i][j])
                    print("bbox",beh_seq_train['bbox'][i][j])
                    print("intention_prob",beh_seq_train['intention_prob'][i][j])
                    print("intention_binary",beh_seq_train['intention_binary'][i][j])
                    print("ped_id",beh_seq_train['ped_id'][i][j])
        print('all images: ',all_img)
def data_balance(data,label_type='intention_binary',sample='OverSample',random_seed=42):
    #gt_labels = [label for gt in seq_data[label_type] for label in gt]
    gt_labels = [gt for gt in data[label_type]]
    num_pos_samples = np.count_nonzero(np.array(gt_labels))
    num_neg_samples = len(gt_labels) - num_pos_samples
   
    print('-----------------Before Balancing--------------------')
    if num_neg_samples == num_pos_samples:
        print('Positive and negative samples are already balanced')
        print('Balanced: \t Positive: {} \t Negative: {} \t Total: {}'.format(num_pos_samples, num_neg_samples,(num_pos_samples+num_neg_samples)))
        return balanced_data
    else:
        print('Unbalanced: \t Positive: {} \t Negative: {}  \t Total: {}'.format(num_pos_samples, num_neg_samples,(num_pos_samples+num_neg_samples)))
    # save the index of the largest class (negative or postive)
    if num_neg_samples > num_pos_samples:
            large_index = np.where(np.array(gt_labels) == 0)[0] # indices of oversampled class
            samll_index = np.where(np.array(gt_labels) == 1)[0] # indices of undersampled class
            diff = num_neg_samples - num_pos_samples
    else:
            large_index = np.where(np.array(gt_labels) == 1)[0] # indices of oversampled class
            samll_index = np.where(np.array(gt_labels) == 0)[0]  # indices of undersampled class
            # index of undersampled class
            diff = num_pos_samples - num_neg_samples
    print('diff----------------------------',diff)     
    if (sample=='OverSample'):
        # randomly select a sample if it is from undersampled class, then add to the dataset
                    # shuffle the indices
        random.seed(random_seed)
        random.shuffle(samll_index)
        # check how many times the samples need to repeat
        if (diff < len(samll_index)): # The samples in the small class need not repeat more than once
            repeated_samples = list(samll_index [0:diff])
        else: # The samples in the large class are more than twice of the small class (assume less than 3 times)
            repeated_samples = list(samll_index)  # all samples are repeated 
            new_diff = diff - len(samll_index)
            additional_repeat = list(samll_index [0:new_diff])
            repeated_samples.extend(additional_repeat)
        list_of_features = []
        for feature in data :# sample classes
            feature_list =[]
            for sample in data[feature]:
                feature_list.append(sample)
            for index in repeated_samples:
                feature_list.append(data[feature][index])
            list_of_features.append(feature_list)

        balanced_data = {'image': list_of_features[0],
                'bbox': list_of_features[1],
                'occlusion': list_of_features[2],
                'intention_prob': list_of_features[3],
                'intention_binary': list_of_features[4],
                'ped_id': list_of_features[5]}


    else: # undersample
        np.random.seed(random_seed)
        np.random.shuffle(large_index)
        selected_images = list(large_index [0:len(samll_index)])
        list_of_features = []
        
        for feature in data :# sample classes
            feature_list =[]
            for index in samll_index:  # add all samples in small class
                feature_list.append(data[feature][index])
            for index in selected_images:  # add large index randomly selected
                feature_list.append(data[feature][index])
            list_of_features.append(feature_list)
        
        balanced_data = {'image': list_of_features[0],
                'bbox': list_of_features[1],
                'occlusion': list_of_features[2],
                'intention_prob': list_of_features[3],
                'intention_binary': list_of_features[4],
                'ped_id': list_of_features[5]}
        
    print('------------------ After Balancing -------------------------')
    gt_labels = [gt for gt in balanced_data[label_type]]
    num_pos_samples = np.count_nonzero(np.array(gt_labels))
    num_neg_samples = len(gt_labels) - num_pos_samples
    if num_neg_samples == num_pos_samples:
        print('Positive and negative samples are now balanced')
        print('Balanced: \t Positive: {} \t Negative: {} \t Total: {}'.format(num_pos_samples, num_neg_samples,(num_pos_samples+num_neg_samples)))
        return balanced_data
    else:
        print('Unbalanced: \t Positive: {} \t Negative: {}  \t Total: {}'.format(num_pos_samples, num_neg_samples,(num_pos_samples+num_neg_samples)))
    
    return balanced_data 

def main():
        imdb = PIE(data_path='/home/avl1/Desktop/Murad/IROS/PIEPredict/From_dataset')
        data_opts = {'fstride': 1,
                'sample_type': 'all', 
                'height_rng': [0, float('inf')],
                'squarify_ratio': 0,
                'data_split_type': 'default',  #  kfold, random, default
                'seq_type': 'intention', #  crossing , intention
                'min_track_size': 0, #  discard tracks that are shorter
                'max_size_observe': 15,  # number of observation frames
                'max_size_predict': 5,  # number of prediction frames
                'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                'balance': True,  # balance the training and testing samples
                'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                'encoder_input_type': [],
                'decoder_input_type': ['bbox'],
                'output_type': ['intention_binary']
                }
        # beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        # beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        # before balancing the data
        print('Initail Data: ')
        data_size(beh_seq_test, print_details=False)
        # Flatten data to handle single images instead of image sequences
        beh_seq_test = flatten(beh_seq_test)
        print('After flattening: ',len(beh_seq_test['intention_binary']))
        beh_seq_test = data_balance(beh_seq_test, label_type='intention_binary',sample='Undersample')

        image_pre_process(beh_seq_test,replace=False,save_shuffled=True)   # replace to replace current data , save shuffled data to save shuffled data
        # train_test_split_data(shuffled=True)

        # # Before training load data into batchs
        # batch_size = 64
        # # loading filenames and labels for training
        # loaded_fname = np.load('FROM_dataset/train_filenames.npy')
        # loaded_labels = np.load('FROM_dataset/labels_train.npy')
        # training_batch_generator = My_Custom_Generator(loaded_fname,loaded_labels,batch_size)    
        # sample = training_batch_generator.__getitem__(0)[0][0] 
        # # lst_str = str(lst)[1:-1]    
        # print('Training number of batches: ' ,len(training_batch_generator), ' Batch size : ',batch_size, 'Image shape: ',sample.shape)#, [length for length in sample.shape()])
        # print('Training Data Shape: ' ,(len(training_batch_generator),batch_size,sample.shape))

if __name__ == '__main__':
    main()