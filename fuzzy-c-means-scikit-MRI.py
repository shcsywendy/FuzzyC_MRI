
# coding: utf-8

# ## Import library

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import os
import cv2
import numpy as np
from time import time
from PIL import Image
import struct
import math

plt.ion()
plt.show()

#get_ipython().magic('matplotlib inline')
#matplotlib.pyplot.ion()

# # Import image
# - a

# In[3]:




'''
format:181*217*181
step:1*1*1
'''
def readImageFromRawb(path):
    res = []
    with open(path,'rb') as file:
        tmp = file.readlines()
        print(tmp.__len__())
        num = 7109137 # size of file
        a = 0  # start position of file
        # show byte by byte
        print(num)
        for c in range(a, num):
            file.seek(c)
            data = struct.unpack("B", file.read(1))
            res.append(data[0])

        print(res.__len__())

    img = np.zeros((181,217,181))
    #img = np.zeros((10,217,181))
    k = 0
    for z in range(181):
        for i in range(217):
            for j in range(181):
                img[z][i][j] = int(res[k])
                k+=1
    return img


# In[4]:

imgs = readImageFromRawb("t1_icbm_normal_1mm_pn3_rf20.rawb")
# img_array=imgs.load()


# ## Function Definition

# In[5]:

def change_color_fuzzycmeans(cluster_membership, clusters):
    img = []
    count=0
    #for pix in cluster_membership.T:
     #   img.append(clusters[np.argmax(pix)]) #clusters=cntr, cluster_membership=u
#       print('img.append=', (clusters[np.argmax(pix)]))
#         print('np=',np.argmax(pix),pix)
#       print('cluster=',clusters[np.argmax(pix)])
#        plt.imshow(img, cmap='gray')
#        plt.show()
#        print('count=',count)
#        count+=1
#    print('T=',len(cluster_membership.T),len(cluster_membership.T[0]))
#    print('count=',count)
    print("cluster_membership.shape=",cluster_membership.shape)
    print("clusters.shape=", clusters.shape)
    cluster_ship = np.argmax(cluster_membership, axis=0)
    print("cluster_menbership.shape=", cluster_ship.shape)
    #print("cluster_menbership.shape=", len(#))
    
    for index in range(0, cluster_membership.shape[1]):
        pixelValue = round(clusters[cluster_ship[index]][0])
        img.append(pixelValue)
    
    return img
   

def readimage():

    list_img = []

    #list_img.append(imgs[45,:,:])
    #list_img.append(imgs[90,:,:])
    #list_img.append(imgs[100,:,:])
    
    return list_img

def readimage1():
    print("imgs.shape=", imgs.shape)
    print("imgs[:,10,:].shape=", imgs[:,10,:].shape)
    list_img = []
    for i in range(50,60):
        list_img.append(imgs[:,i,:])
    return list_img

def read_acc_image1():
    list_img = []
    for i in range(60,70):
        list_img.append(acc_result[:,i,:])
    return list_img

def readimage2():
    list_img = []
    for i in range(217):
        list_img.append(imgs[:,i,:])
    return list_img

def readimage3():
    list_img = []
    for i in range(181):
        list_img.append(imgs[:,:,i])
    return list_img

def bwarea(img):
    row = img.shape[0]
    col = img.shape[1]
    total = 0.0
    for r in range(row-1):
        for c in range(col-1):
            sub_total = img[r:r+2, c:c+2].mean()
            if sub_total == 255:
                total += 1
            elif sub_total == (255.0/3.0):
                total += (7.0/8.0)
            elif sub_total == (255.0/4.0):
                total += 0.25
            elif sub_total == 0:
                total += 0
            else:
                r1c1 = img[r,c]
                r1c2 = img[r,c+1]
                r2c1 = img[r+1,c]
                r2c2 = img[r+1,c+1]
                
                if (((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1)):
                    total += 0.75
                else:
                    total += 0.5
    return total
            
def imclearborder(imgBW):

    # Given a black and white image, first find all of its contours
    radius = 2
    imgBWcopy = imgBW.copy()
    image, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye! (rowCnt >= imgRows-1-radius and rowCnt < imgRows)保护边界
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0), -1)

    return imgBWcopy

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    image, contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0), -1)

    return imgBWcopy      

def imfill(im_th):
    
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    
    return im_out


# In[6]:

def Ave(img, bwfim3,ii,jj,n,k):
    flag = False 
    for j in range(jj,jj+n,1):
        for i in range(ii,ii+n,1):
             if (j<k*i<j+1) or (j<k*(i+1)<j+1):
                    flag= True
    if flag:  
        a=0
        ave=0
        for j in range(jj,jj+n,1):
            for i in range(ii,ii+n,1):
                a+=img[j][i]
        ave=a/(n*n)
        for j in range(jj,jj+n,1):
            for i in range(ii,ii+n,1):
#                 print(ave)
                bwfim3[j][i]=ave


# In[7]:

def Ave2(img,bwfim3,x,y,BL,BH):
    line=0
    line=cutline(img,x,y,BL,BH)
    #print('b=',b)
    for i in range(x,x+BH,1):
         for j in range(y,y+BL,1):
            if (img[i][j]>=line):
                    bwfim3[i][j]=255
            elif (img[i][j]<line):
                    bwfim3[i][j]=0

    


# In[8]:

def Ave3(img,bwfim3,x,y,BL,BH):
    line=0
    #line=cutline(img,x,y,BL,BH)
    line=100
    #print('b=',b)
    for i in range(x,x+BH-1,1):
         for j in range(y,y+BL-1,1):
            if (img[i][j]>=line-3):
                    bwfim3[i][j]=0
            elif ():
                    bwfim3[i][j]=255


# In[9]:

def cutline(img,x,y,BL,BH):
    a=0
    count=0
    for i in range(x,x+BH,1):
        for j in range(y,y+BL,1):
                if (img[i][j]<12):
                    count+=1
                elif(img[i][j]>=12):
                    a+=img[i][j]
                
    
    line=a/(((BL*BH)-count)+1)
    return line


# In[10]:

def clear(img):
     for i in range(len(img)):
            for j in range(len(img[0])):
                img[i][j]=0


# In[11]:

# a=0
# b=0
#while a<3:
#     b=0
#     while b<3:
#         print(a,b)
#         b += 1


# In[14]:

def getColorImage(u,cntr,rows, cols):
    cluster_ship = np.argmax(u, axis=0)
    #img = np.zeros((rows, cols), np.float)
    img = np.zeros((rows, cols), np.float64)  # Use np.float64 to explicitly use NumPy's 64-bit float

    colorLevel = cntr.shape[0]
    for index in range(0, u.shape[1]):
        y = index // cols
        x = index % cols
        img[y][x] = round(255. * cluster_ship[index] / (colorLevel - 1)) 
    return img


# ## Process

# In[22]:

my_result = []
list_img = readimage1()
n_data = len(list_img)
clusters = [80]

print(n_data)

for picIndex in range(6, 8):
    rgb_img = list_img[picIndex]
    print("===============================================================")
    print(rgb_img.shape)
    img = np.reshape(rgb_img, rgb_img.shape).astype(np.uint8)
    shape = np.shape(img)
    
    #print('shape=',shape)
    # initialize graph
    plt.figure(figsize=(20,20))
    plt.subplot(1,4,1)

    print('Image '+str(picIndex+1))
    for t,cluster in enumerate(clusters):
            
        # Fuzzy C Means
        new_time = time()
        
        
        print("******************************************************************************")
        print(rgb_img.T.shape)
        
        print("******************************************************************************")
        
        rgb__img = rgb_img.reshape(-1, 1)
        
        #rgb_img = np.zeros((shape[0] * shape[1], 3), np.float)
        rgb_img = np.zeros((shape[0] * shape[1], 3), np.float64)  
        for index in range(0, rgb__img.shape[0]):
            y = int(index) // shape[1]
            x = int(index) % shape[1]
            rgb_img[index][0] =  rgb__img[index]
            rgb_img[index][1] = x * 0.5 # 
            rgb_img[index][2] = y * 0.5 # 
            
        print(rgb_img[4:16, :])
        #exit()
        
        print("rgb_img.shape=", rgb_img.shape)
        

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        rgb_img.T, cluster, 1.35, error=0.0001, maxiter=40, init=None,seed=42)
        
        new_img = change_color_fuzzycmeans(u,cntr)
       
        fuzzy_img = np.reshape(new_img,shape).astype(np.uint8)
       
        colorImage = getColorImage(u,cntr, shape[0], shape[1])
        
        #colorImage1 = np.reshape(colorImage,shape).astype(np.uint8)
        colorImage1 = np.reshape(colorImage,shape).astype(np.uint8)
    
        plt.imshow(fuzzy_img, cmap='gray')
        plt.show()
   
        plt.imshow(img, cmap='gray')
        plt.show()
        
        plt.imshow(colorImage1, cmap='gray')
        plt.show()


# ## acc

# In[23]:

acc_result = readImageFromRawb('phantom_1.0mm_normal_gry.rawb')


# In[24]:

list_acc_img = read_acc_image1()
for i in range(10):
    img1 =list_acc_img[i]
    img2=list_img[i]
    print('i=',i)
    print('accurate result is showed here')
    plt.imshow(img1, cmap='gray')
    plt.show()
    print('my result is showed here')
    plt.imshow(img2, cmap='gray')
    plt.show()
    
    
# img2 = my_result[1]
# print('accurate result is showed here')
# plt.imshow(img1, cmap='gray')
# plt.show()
# print('my result is showed here')
# plt.imshow(img2, cmap='gray')
# plt.show()


# In[25]:

# answer_list = [1,2,3,4,5,6,7,8,9,10]
for k in range(len(my_result)):
    img1 = list_acc_img[k]
    img2 = my_result[k]
    plt.imshow(img1, cmap='gray')
    plt.show()
    plt.imshow(img2, cmap='gray')
    plt.show()
    h,w = img1.shape
    allpoint = 0
    accpoint = 0
    for i in range(h):
        for j in range(w):
            if img1[i,j] > 0 :
                allpoint += 1
                if img2[i,j]>0:
                    accpoint += 1
    acc = accpoint/(allpoint+1)
    print('acc=',acc)





