import numpy as np
import os
import cv2

modes = ['01deeplabv3', '02scselink', '03unet', '04unetpp', '05attenunet',
         '06abcnet', '07banet', '08unetformer', '09dcswin', '10dlinkvit_h8', '11vit']


path = '/data3/Seafog_HBH/seafog_year_0412/val_2021/'
dates = ['20210325_0000.npy', '20210613_0300.npy',
         '20210122_0030.npy', '20210212_0730.npy',
         '20210509_0600.npy', '20210510_0300.npy',
         '20210512_0130.npy']
# dates = [x for x in os.listdir(path) if x[-4:]=='.npy']
print(len(dates))

for date in dates:
    print(date)
    data = np.load(path+date)

    rgbimg = cv2.resize(cv2.merge((data[:,:,0:1],data[:,:,1:2],data[:,:,2:3])),(256,256))
    label = cv2.resize(cv2.imread(path+'conn_'+date.split('.')[0]+'_300_sp.png'),(256,256))
    cv2.imwrite('/data3/Seafog_HBH/cloud/pred_vis0524/down/'+date.split('.')[0]+'.png', rgbimg)
    cv2.imwrite('/data3/Seafog_HBH/cloud/pred_vis0524/down/'+date.split('.')[0]+'_label.png', label)
    
    # merge_data = np.zeros(shape=(2*256,7*256,3))
    # merge_data[:256,:256,:] = rgbimg
    # merge_data[256:,:256,:] = label
    
    names = locals()
    
    for i, mode in enumerate(modes):
        predpath = '/data3/Seafog_HBH/cloud/pred_final0524/'+mode+'/'+date.split('.')[0]+'.png'
        names['img'+mode] = cv2.resize(cv2.imread(predpath),(256,256))
        
        cv2.imwrite('/data3/Seafog_HBH/cloud/pred_vis0524/down/'+date.split('.')[0]+'_'+mode+'.png', names['img'+mode])
        

    # merge_data[:256,256:512,:] = img01
    # merge_data[:256,512:768,:] = img02
    # merge_data[:256,768:1024,:] = img03
    # merge_data[:256,1024:1280,:] = img04
    # merge_data[:256,1280:1536,:] = img05
    # merge_data[:256,1536:,:] = img06
    
    # merge_data[256:,256:512,:] = img07
    # merge_data[256:,512:768,:] = img08
    # merge_data[256:,768:1024,:] = img09
    # merge_data[256:,1024:1280,:] = img10
    # merge_data[256:,1280:1536,:] = img11
    # # merge_data[256:,1536:,:] = img11
    
    # cv2.imwrite('/data3/Seafog_HBH/cloud/pred_vis0524/'+date.split('.')[0]+'.png', merge_data)