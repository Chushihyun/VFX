import math
import numpy as np
from numpy.linalg import inv
import cv2
from copy import deepcopy
from tqdm import tqdm,trange
import random

def window_sum(img_error):
    window_error=np.zeros(img_error.shape)
    r=2

    for i in range(img_error.shape[0]):
        try:
            for j in range(img_error.shape[1]):
                for k in range(img_error.shape[2]):
                    s=np.sum(img_error[i,j-r:j+r+1,k-r:k+r+1])
                    window_error[i][j][k]=s//((2*r+1)**2)
        except:
            continue

    return window_error


def local_max(img):
    r=3
    t=100
    tolerance = 10
    feature=[]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                tmp=img[i-r:i+r+1,j-r:j+r+1]
                # print(tmp)
                if img[i][j]==tmp.max() and img[i][j]>t:
                    if 2*tolerance<i<img.shape[0]-2*tolerance and 2*tolerance<j<img.shape[1]-2*tolerance:
                        feature.append([i,j])
            except:
                continue
    return feature
    


def single_feature_detection(img):

    h,w=img.shape[0],img.shape[1]
    img_original=deepcopy(img).astype(np.int)

    shift=[[1,0],[1,1],[0,1],[-1,1]]
    imgs_shift=[]
    for s in shift:
        img_s=np.roll(img,s[0],axis=0)
        img_s=np.roll(img_s,s[1],axis=1)
        imgs_shift.append(deepcopy(img_s))

    imgs_shift=np.array(imgs_shift).astype(np.int)
    imgs_error=np.zeros(imgs_shift.shape).astype(np.int)

    for s in range(len(shift)):
        imgs_error[s]=(img_original-imgs_shift[s])**2

    window_error=window_sum(imgs_error)
    min_error=np.min(window_error,axis=0)
    feature_points=local_max(min_error)

    return feature_points

def draw_points(img, points):
    print(img.shape)
    new_image=deepcopy(img)
    color = (255, 255, 0)
    for p in points:
        cv2.circle(new_image, tuple(p[::-1]), 2, color, -1)

    #cv2.imwrite('match2/point.jpg', new_image)

    #cv2.imshow("new", new_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return

def feature_descriptor(img,features):
    s=7
    feature_description=[]
    for x,y in features:
        window=img[x-s:x+s,y-s:y+s]
        feat=np.array(window).flatten()
        feature_description.append(feat)

    return np.array(feature_description)

                

def feature_detection(img_list):
    feature_list=[]
    for img in img_list:
        image_feature=single_feature_detection(img)
        draw_points(img, image_feature)

        description=feature_descriptor(img,image_feature)
        print(description.shape)
        feature_list.append([image_feature,description])
    
    return feature_list
    
    # sift = cv2.SIFT_create()
    # for img in img_list:
    #     kp, des = sift.detectAndCompute(img, None)
    #     feature_list.append([kp,des])

    # return feature_list

def projection(images,focal_lengths):
    images_cylindrical = np.zeros(images.shape, dtype=np.uint8)
    sphere = min(focal_lengths)
    for i, img in enumerate(images):
        f = focal_lengths[i]
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                y_normal = (y-(img.shape[0]/2)) / sphere * ((x-(img.shape[1]/2))**2 + f**2)**0.5
                y_normal += img.shape[0]/2
                x_normal = f * math.tan((x-(img.shape[1]/2)) / sphere)
                x_normal += img.shape[1]/2
                if int(y_normal) not in range(0, img.shape[0]) or int(x_normal) not in range(0, img.shape[1]):
                    continue
                images_cylindrical[i][y][x] = img[int(y_normal)][int(x_normal)]
        #cv2.imwrite("cylindrical/img{}.jpg".format(str(i)), images_cylindrical[i])

    return images_cylindrical


def feature_matching(i, feat1,feat2,img1,img2):
    kp1, des1=feat1
    kp2, des2=feat2

    good_matches=[]
    
    
    for idx1, f1 in enumerate(des1):
        f1 = np.repeat([f1], len(des2), axis=0)
        diff = np.power((des2 - f1), 2).sum(axis = 1)
        diff1, diff2 = np.partition(diff, 2)[0:2]
        if diff1 / diff2 < 0.8:
            good_matches.append(cv2.DMatch(idx1, np.argmin(diff),0, float(diff1)))
            #matches.append([idx1, np.argmin(diff), d1])
    good_matches =sorted(good_matches, key=lambda x:x.distance)
    
    # matcher = cv2.BFMatcher()
    # matches = matcher.knnMatch(des1, des2, k=2)
    # good_matches = []
    # num = 20
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)

    # good_matches = sorted(good_matches, key=lambda x: x.distance)
    # good_matches = good_matches[:num]

    # points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    # points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    points1 = np.array([kp1[m.queryIdx] for m in good_matches])
    points2 = np.array([kp2[m.trainIdx] for m in good_matches])
    
    keypoints1=[]
    keypoints2=[]
    for point in kp1:
        keypoints1.append(cv2.KeyPoint(point[1],point[0],1))
    for point in kp2:
        keypoints2.append(cv2.KeyPoint(point[1],point[0],1))
    img_draw_match = cv2.drawMatches(
       img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)

    # img_draw_match = cv2.drawMatches(
    #    img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)

    #cv2.imwrite('match2/{}.jpg'.format(str(i)), img_draw_match)
    #print('match2/{}.jpg saved'.format(str(i)))


    
    # img_draw_match = cv2.drawMatches(
    #    img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_DEFAULT)
    # cv2.imwrite('match/{}.jpg'.format(str(i)), img_draw_match)
    
    # cv2.namedWindow('match',0)
    # cv2.resizeWindow('match', 1200, 800)
    # cv2.imshow('match', img_draw_match)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return points1,points2
    
    

def fit(p_source, p_target):
    # shift=np.array([[1,0,0],[0,1,0],[0,0,1]])
    # x_mean=np.mean(p_source[:,0])-np.mean(p_target[:,0])
    # y_mean=np.mean(p_source[:,1])-np.mean(p_target[:,1])
    # shift[0][2]=-x_mean
    # shift[1][2]=-y_mean
    # return shift


    A = np.zeros((2 * len(p_source), 6))
    for i in range(p_source.shape[0]):
        A[2*i][0:2] = p_source[i]
        A[2*i][2] = 1
        A[2*i+1][3:5] = p_source[i]
        A[2*i+1][5] = 1
    b = p_target.reshape(-1)
    ATA = np.matmul(A.T, A)
    try:
        shift = np.matmul(inv(ATA), A.T)
    except:
        shift = np.matmul(ATA, A.T)
    shift = np.matmul(shift, b)
    shift = shift.reshape(2, 3)
    shift = np.concatenate((shift, np.array([[0, 0, 1]])), axis=0)
    return shift
    

def transform(point,shift):
    point = np.concatenate((point, np.array([1])), axis=0)
    return np.matmul(shift, point.T)[:2] 

def calculate_inlier(points1,points2,shift):
    num_points=len(points1)
    inlier=0
    error_tolerate=8

    # may be accelerate by numpy
    for i in range(num_points):
        new_point=transform(points1[i],shift)
        if np.linalg.norm(new_point-points2[i],2)<error_tolerate:
            inlier+=1
    
    return inlier


def calculate_shift(points1,points2):
    # points1[:, [0, 1]] = points1[:, [1, 0]]
    # points2[:, [0, 1]] = points2[:, [1, 0]]
    num_points=len(points1)
    ransac_time=1000
    ransac_point_num=6
    best_inlier=0

    for i in range(ransac_time):
        idx_list=random.sample(range(num_points),ransac_point_num)
        shift = fit(points1[idx_list], points2[idx_list])
        inlier=calculate_inlier(points1,points2,shift)
        
        if inlier>best_inlier:
            best_inlier=inlier
            best_shift=shift
    print(f"inlier: {best_inlier}/{num_points}")
    print(best_shift)
    return best_shift


def image_stitching(feature_list,images):
    shifts=[]
    # feature matching
    for i in range(1, len(feature_list)):
        points1,points2=feature_matching(i, feature_list[i-1],feature_list[i],images[i-1],images[i])
        s=calculate_shift(points1,points2)
        shifts.append(s)
    # blending
    result=image_blending(images,shifts)

    return result,shifts


def load_param(path):
    file = open(path)
    lines = file.readlines()
    focal_lengths = []
    for i in range(11, len(lines), 13):
        focal_lengths.append(float(lines[i]))
    return focal_lengths

def boundary(img,H, offset_x):
    corners = np.array([[0, 0, 1],
                    [0, img.shape[1], 1],
                    [img.shape[0] - offset_x, 0, 1],
                    [img.shape[0] - offset_x, img.shape[1], 1]])
    corners = H @ corners.T
    corners = (corners/corners[-1])[0:2]
    min_x = corners[0].min()
    max_x = corners[0].max()
    min_y = corners[1].min()
    max_y = corners[1].max()
    return int(min_x), int(max_x), int(min_y), int(max_y)
    

def single_image_blending(img1,img2,H,offset_x):

    offset_x=0
    x1, x2, y1, y2 = boundary(img1, H, offset_x)
    max_x = max(x2, img2.shape[0])
    max_y = max(y2, img2.shape[1])
    min_x = min(x1, 0)
    min_y = 0

    result = np.zeros((max_x - min_x, y2, 3), dtype = np.uint8)
    print(result.shape, x1, x2, y1, y2)
    result[-min_x + offset_x : -min_x + offset_x + img2.shape[0], 0: img2.shape[1]]= img2
    
    H_inv = np.linalg.inv(H)
    h ,w, _ = img1.shape
    for i in range (result.shape[1]):
        for j in range (result.shape[0]):
            pos = H_inv @ [[j], [i], [1]]
            x, y = (pos/pos[-1])[0:-1]
            int_x = int(x)
            int_y = int(y)
            #blending
            if (x >= 0) and (x < h) and (y >= 0) and (y < w):
                if img1[int_x][int_y].all() == 0:
                    continue
                if result[j][i].all() != 0:
                    # y
                    
                    dy1 = img2.shape[1] - i
                    dy2 = i - y1
                    
                    # x
                    #dx1 = min(h - x, x - 0)
                    #dx2 = min(j - (-min_x), (img2.shape[0] - min_x) - j)
                    #dx2 = min(j , (img2.shape[0]) - j)
                    
                    #rx1 = dx1 / (dx1 + dx2)
                    w1 = dy1 / (dy1 + dy2)
                    w2 = 1 - w1
                    # bilinear
                    #w1 = ry1 / (ry1 + ry2)
                    #w2 = ry2 / (ry1 + ry2)
                    
                    # linear
                    # w1 = dx1 / (dx1 + dx2)
                    # w2 = dx2 / (dx1 + dx2)
                    result[j][i] = (w2 * img1[int_x][int_y] + w1 * result[j][i])
                    
                else:
                    result[j][i] = img1[int_x][int_y]
    return result, min_x

def image_blending(images,shifts):
    # print(shifts)
    num_image = len(images)
    h = images[0].shape[0]
    w = images[0].shape[1]
    new = np.zeros((h*3, w * num_image, 3))
    start = [h, 0]
    #new[start[0]: start[0]+h, start[1]: start[1]+w] = images[-1]
    shift = np.identity(3)
    result = images[-1]
    offset = 0
    for i in range(len(images)-2, 0, -1):
        image = images[i]
        #shift = np.matmul(shifts[i], shift)
        shift = shift @ shifts[i]
        print(f"blending the {i} image")
        result, offset = single_image_blending(image, result, shift, offset)
        #cv2.imwrite('blend1/{}.jpg'.format(str(i)), result)
    #cv2.imwrite('blend1/final.jpg', result)
     
    # ipdb.set_trace()


    return result
