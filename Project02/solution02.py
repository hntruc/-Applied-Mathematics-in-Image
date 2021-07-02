import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def data(path):
    img = Image.open(path).convert('RGB')
    img_np = np.asarray(img)# convert to numpy array
    return img_np

def change_light(img_npa, scalar):
    temp = img_npa.copy()
    for ele in temp:
        for ele_child in ele:
            for pos in range(3):
                if (ele_child[pos] + scalar) > 255:
                    ele_child[pos] = 255
                else:
                    ele_child[pos] += scalar
                    
    return temp

def change_contrast(img_npa, scalar):
    temp = img_npa.copy()
    for ele in temp:
        for ele_child in ele:
            for pos in range(3):
                if (ele_child[pos] * scalar) > 255:
                    ele_child[pos] = 255
                else:
                    ele_child[pos] *= scalar
                    
    return temp

def change_grey(img_npa):
    temp = img_npa.copy()
    for ele in temp:
        for ele_child in ele:
            value = ele_child[0]*0.3 + ele_child[1]*0.59 + ele_child[2]*0.11
            for pos in range(3):
                ele_child[pos] = value
    return temp

def stack_photo(img1, img2):
    shape = np.array(img1).shape
    width = shape[0]
    height = shape[1]
    img1 = np.reshape(img1,(width*height, 3))
    img2 = np.reshape(img2,(width*height, 3))
    ans = []
    for i in range(width*height):
        ans.append((img1[i]+img2[i])//2)
    return ans, width, height

def vertical(img_npa):
    ans = []
    shape = np.array(img_npa).shape
    width = shape[0]
    height = shape[1]
    img = np.reshape(img_npa,(width*height, 3))
    for i in range((width*height)):
        ans.append(img[width*height - i -1])
    return ans, width, height

def horizontal(img_npa):
    re = []
    for ele in img_npa:
        ans = []
        for e in range(len(ele)-1, -1, -1):
            ans.append(ele[e])
        re.append(ans)
    return re

def cal_avg(img_npa, y, x): #y is i, x is j
    shape = np.array(img_npa).shape
    w = shape[0]
    h = shape[1]
    c, r = [], []

    for i in np.arange(-1,1):
        if x + i < 0:
            c.append(0)
        elif x+i > w - 1:
            c.append(w - 1)
        else: 
            c.append(x + i)

    for i in np.arange(-1,1):
        if y + i < 0:
            r.append(0)
        elif y+i > h-1:
            r.append(h - 1)
        else:
            r.append(y + i)

    sum = np.zeros(3)
    for i in r:
        for j in c:
            sum += img_npa[i][j]/9
    
    sum = [int(i) for i in sum]
    return sum

def blur(img_npa):
    img_temp = img_npa.copy()
    for pos,ele in enumerate(img_temp):
        for e in range(len(ele)):
            temp = cal_avg(img_npa,pos,e)
            ele[e] = temp
    return img_npa
    
def main():
    print('1. Thay đổi độ sáng cho ảnh \n2. Thay đổi độ tương phản \n3. Chuyển đổi ảnh RGB thành ảnh xám \n4. Chồng 2 ảnh cùng kích thước \n5. Xoay dọc ảnh \n6. Xoay ngang ảnh \n7. Làm mờ ảnh')
    choice = int(input('Nhập lựa chọn: '))
    if choice in [1,2,3,4,5,6,7]:
        if choice == 1:
            path = input('Nhập link hình ảnh: ')
            scalar = int(input('Nhập mức độ sáng(scalar): '))
            img_np = data(path)
            img_npa = change_light(img_np, scalar) 
            imgplot = plt.imshow(img_npa)
            plt.show()
        elif choice == 2:
            path = input('Nhập link hình ảnh: ')
            scalar = int(input('Nhập mức độ tương phản(scalar): '))
            img_np = data(path)
            img_npa = change_contrast(img_np, scalar)
            imgplot = plt.imshow(img_npa)
            plt.show()
        elif choice == 3:
            path = input('Nhập link hình ảnh: ')
            img_np = data(path)
            img_npa = change_grey(img_np)
            imgplot = plt.imshow(img_npa)
            plt.show()
        elif choice == 4:
            path1 = input('Nhập link ảnh 1: ')
            img1 = data(path1)
            path2 = input('Nhập link ảnh 2: ')
            img2 = data(path2)
            img_npa, w, h = stack_photo(img1, img2)
            img_npa = np.reshape(img_npa, (w,h,3))
            imgplot = plt.imshow(img_npa)
            plt.show()
        elif choice == 5:
            path = input('Nhập link hình ảnh: ')
            img_np = data(path)
            img_npa, w, h = vertical(img_np)
            img_npa = np.reshape(img_npa, (w,h,3))
            imgplot = plt.imshow(img_npa)
            plt.show()
        elif choice == 6:
            path = input('Nhập link hình ảnh: ')
            img_np = data(path)
            img_npa = horizontal(img_np)
            imgplot = plt.imshow(img_npa)
            plt.show()
        elif choice == 7:
            path = input('Nhập link hình ảnh: ')
            img_np = data(path)
            img_npa = blur(img_np)
            imgplot = plt.imshow(img_npa)
            plt.show()

main()