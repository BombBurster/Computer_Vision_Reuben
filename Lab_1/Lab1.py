import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pandas as pd
import numpy as np

files = glob.glob('./Data/*.png')
images = []
for file in files:
    images.append(mpimg.imread(file))

landmarks = pd.read_csv('./Data/landmarks.csv')

coord = np.empty((0, 3), dtype=np.float)
cols = landmarks.columns.size
i = 0
while i < cols:
    if i is not 0:
        coord = np.row_stack((coord, np.array([landmarks.iloc[0, i], landmarks.iloc[0, i+1], 1])))
        i = i+2
    else:
        i = i+1

# plot the image with the reference points
plt.figure()
plt.axis('off')
imgplt = plt.imshow(images[1])
plt.scatter(x=coord[:, 0], y=coord[:, 1], marker='x', c='b', s=20)
plt.show()

# ref = np.empty((6, 3), dtype=np.float)
refx = np.array([30, 98, 64, 48, 64, 80], dtype=np.float)
refy = np.array([30, 30, 64, 98, 98, 98], dtype=np.float)
ones = np.ones((6,), dtype=np.float)
ref = np.column_stack((refx, refy, ones))

plt.figure()
plt.axis('off')
black = np.ones((128, 128), dtype=np.float)
imgplt = plt.imshow(black , cmap='gray')
plt.scatter(x=ref[:,0], y=ref[:,1], marker='x', c='r', s=20)
plt.show()

coordT = coord.T
# print(coordT)
eq_p_1 = coordT.dot(coord)
eq_p_1 = np.linalg.inv(eq_p_1)
eq_x = coordT.dot(ref[:, 0])
eq_y = coordT.dot(ref[:, 1])
eq_x = eq_p_1.dot(eq_x)
eq_y = eq_p_1.dot(eq_y)
T3 = np.array([0, 0, 1], dtype=np.float)
Trans_Matrix = np.column_stack((eq_x, eq_y, T3))
print(Trans_Matrix)

new_Coord = coord.dot(Trans_Matrix)
plt.figure()
plt.axis('off')
black = np.ones((128, 128), dtype=np.float)
imgplt = plt.imshow(black , cmap='gray')
plt.scatter(x=ref[:, 0], y=ref[:, 1], marker='x', c='r', s=20)
plt.scatter(x=new_Coord[:, 0], y=new_Coord[:, 1], marker='x', c='b', s=20)
plt.show()

im = images[1]
# im2 = []

Tinv = np.linalg.pinv(Trans_Matrix)
im2 = np.zeros((128, 128, 4))
for x in range(128):
    for y in range(128):
        # Derive the pixel coordinates to take from the reference image
        v, w, _ = np.dot([x, y, 1], Tinv)
        # print(v, w)
        v = np.round(v).astype('int')
        w = np.round(w).astype('int')
        im2[y, x, :] = im[w, v, :]
plt.figure()
plt.imshow(im2)
plt.axis('off')
plt.scatter(new_Coord[:, 0], new_Coord[:, 1], marker="x", color='blue')
plt.savefig('Result/Thor.png')
plt.show()
