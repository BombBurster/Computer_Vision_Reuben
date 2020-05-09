import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

img = cv2.imread('checkerboard.jpg', 0)
plt.figure()
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# import pickle

img_pts_file = open('image_points.pkl','rb')
obg_pts_file = open('object_points.pkl', 'rb')
img_pts = pickle.load(img_pts_file)
obg_pts = pickle.load(obg_pts_file)
# img_pts = pd.read_pickle('image_points.pkl')
# obg_pts = pd.read_pickle('object_points.pkl')
print(img_pts)
print(obg_pts)

plt.figure()
plt.imshow(img, cmap='gray')
plt.scatter(x=img_pts[:, 0], y=img_pts[:, 1], c='b', marker='x', s=20)
plt.axis('off')
plt.show()

b = []
for i in range(len(img_pts)):
    b.append([img_pts[i,0]])
    b.append([img_pts[i,1]])

b = np.array(b)
print(b)

# A1 = np.zeros((2*len(obg_pts), 6), dtype=np.float)
# X = obg_pts[:, 0]
# Y = obg_pts[:, 1]
# A1[::2,0] = X
# A1[::2,1] = Y
# A1[::2,2] = 1
# A1[1::2,3] = X
# A1[1::2,4] = Y
# A1[1::2,5] = 1


A = np.empty((0, 6),dtype=np.float)
j = 0
for i in range(2*len(obg_pts)):
   if i%2 == 0:
       A = np.row_stack((A, np.array([obg_pts[j,0], obg_pts[j, 1], 1, 0, 0, 0])))   #((A, np.array([obg_pts[:,0], obg_pts[:,1], 1, 0, 0, 0])))
       # A = np.concatenate(A, np.array([obg_pts[:,0], obg_pts[:,1], 1, 0, 0, 0]), axis=0)
       # A.append(np.array([obg_pts[:,0], obg_pts[:,1], 1, 0, 0, 0])) #np.row_stack((A, np.array([obg_pts[:,0], obg_pts[:,1], 1, 0, 0, 0])))
   elif i%2 != 0:
       A = np.row_stack((A, np.array([0, 0, 0, obg_pts[j, 0], obg_pts[j, 1], 1])))
       j = j+1
       #A = np.vstack((A, np.array([0, 0, 0, obg_pts[:,0], obg_pts[:,1], 1])))
       # A.append(np.array([0, 0, 0, obg_pts[:,0], obg_pts[:,1], 1])) #np.row_stack((A, np.array([0, 0, 0, obg_pts[:,0], obg_pts[:,1], 1])))
       #A = np.concatenate(A, np.array([0, 0, 0, obg_pts[:,0], obg_pts[:,1], 1]), axis=0)
#
#A = np.array(A)
print(A)
# print(A1)

# AT = A.T
# prod_inv = AT.dot(A)
# prod_inv = np.linalg.pinv(prod_inv)
# H_o = AT.dot(b)
# H_o = prod_inv.dot(H_o)
#H = np.array((np.array([H_o[0],H_o[1],H_o[2]]),np.array([H_o[3],H_o[4],H_o[5]]),np.array([0,0,1]))

# print(H)