import cv2
import os
import numpy as np


#20180509_0930.png 
# /data3/seafog
# 2,3,13

if __name__=='__main__':
    filename="20180509_0930.npy"
    for filename in os.listdir("/data3/seafog/train"):
        if(not filename.endswith("npy")): continue
        date=int(filename.split('_')[-1].split('.')[0])
        if(date>=800 and date <= 930):
            pass
        else:
            continue
        path=os.path.join("/data3/seafog/train",filename)
        data=np.load(path)
        B=data[:,:,2]
        G=data[:,:,3]
        R=data[:,:,13]
        img=cv2.merge([B,G,R])
        # cv2.imwrite("./prepare/test.png",img)

        img_r = R        #cv2.equalizeHist(R)
        img_g = cv2.equalizeHist(np.array(G))
        img_b = cv2.equalizeHist(B)
        img_heb=cv2.merge([img_b,img_g,img_r])
        # cv2.imwrite("./prepare/test2.png",img_heb)

        total=np.zeros((1024,2048,3),np.uint8)
        total[:,:1024,:]=img
        total[:,1024:,:]=img_heb
        cv2.imwrite("./prepare/preview/{}".format(filename.replace("npy","png")),total)
    pass