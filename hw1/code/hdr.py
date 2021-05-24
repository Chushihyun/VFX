import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

class HDR():
    def __init__(self, imgs, time):
        self.imgs = imgs    #n * h * w * c
        self.time = time
        self.radiance_map = []
        self.g = [] 
    def calculate(self):
        for channel in range(self.imgs.shape[-1]):
            img_gray = self.imgs[:, :, :, channel]
            radiance_map, g = self.single_hdr(img_gray)
            #radiance_map = np.expand_dims(radiance_map, 2)
            self.radiance_map.append(radiance_map)
            self.g.append(g)
        self.radiance_map = np.stack(self.radiance_map, axis=2)
    def single_hdr(self, img_gray):
        # img_gray represent single channel
        num_image=img_gray.shape[0]
        h = 20
        w = 50
        num_points = 0
        position = []
        h_step = img_gray[0].shape[0]/h
        w_step = img_gray[0].shape[1]/w
        for i in range(h-1):
            for j in range(w-1):
                position.append((int(h_step*(i+1)), int(w_step*(j+1))))
                num_points += 1

        value_seq = []
        for pos in position:
            seq = []
            for i in range(img_gray.shape[0]):
                seq.append(img_gray[i][pos[0]][pos[1]])
            value_seq.append(seq)

        A = np.zeros((num_image*num_points+1+254, 256+num_points), dtype=float)
        b = np.zeros((num_image*num_points+1+254))
        for i in range(num_points):
            for j in range(num_image):
                A[i*num_image+j][value_seq[i][j]] = 1
                A[i*num_image+j][256+i] = -1
                b[i*num_image+j] = float(math.log(self.time[j]))
        A[num_image*num_points][128] = 1
        for i in range(254):
            A[num_image*num_points+1+i][i:i+3] = [1, -2, 1]

        result,_,_,_ = np.linalg.lstsq(A, b, rcond=None)

        g = result[:256]
        ln_e = result[256:]

        ln_e_all=np.zeros((np.array(img_gray)[0].shape))
        for i in range(ln_e_all.shape[0]):
            for j in range(ln_e_all.shape[1]):
                weight=0.00001
                total=0
                for p in range(num_image):
                    value=img_gray[p][i][j]
                    w=self.weight_calculate(value)
                    total+=w * (g[value]-math.log(self.time[p]))
                    weight+=w
                ln_e_all[i][j]=total/weight
        radiance_map=copy.deepcopy(ln_e_all)
        #print(radiance_map)

        #radiance_map=radiance_map-np.mean(radiance_map)/np.std(radiance_map)
        new_radiance_map=np.exp(radiance_map)
        return new_radiance_map, g
    def weight_calculate(self, x):
        if x<128:
            return x
        elif x>=128:
            return 256-x
if __name__ == '__main__':
    img1 = cv2.imread('img1.jpg')
    img2 = cv2.imread('img2.jpg')
    img3 = cv2.imread('img3.jpg')
    img4 = cv2.imread('img4.jpg')
    img5 = cv2.imread('img5.jpg')

    print(img1.shape)

    img1=cv2.resize(img1,(576,384))
    img2=cv2.resize(img2,(576,384))
    img3=cv2.resize(img3,(576,384))
    img4=cv2.resize(img4,(576,384))
    img5=cv2.resize(img5,(576,384))

    print(img1.shape)

    # %% [markdown]
    # Image alignment

    # %%


    """
    img_gray = []
    img_gray.append(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    img_gray.append(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    img_gray.append(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY))
    img_gray.append(cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY))
    img_gray.append(cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY))
    img_gray = np.array(img_gray)

    num_image = img_gray.shape[0]

    print(img_gray.shape)
    """
    """
    #plt.figure(figsize=(16,16))

    for i in range(5):
        print(i)
        m=np.median(img_gray[i])
        img_gray[i][img_gray[i]>=m]=1
        img_gray[i][img_gray[i]<m]=0
        #plt.subplot(3,2,i+1)
        #plt.imshow(img_gray[i],cmap='gray')
    #plt.show()

    img=img_gray[4]-img_gray[0]
    print(img.shape)
    for i in img:
        for j in i:
            if j!=0:
                print("yes")
    print(abs(img.sum()))
    """


    # %%


    # %%
    img_b=[]
    img_b.append(img1[:,:,0])
    img_b.append(img2[:,:,0])
    img_b.append(img3[:,:,0])
    img_b.append(img4[:,:,0])
    img_b.append(img5[:,:,0])
    img_b=np.array(img_b)
    #print(img_b)

    img_g=[]
    img_g.append(img1[:,:,1])
    img_g.append(img2[:,:,1])
    img_g.append(img3[:,:,1])
    img_g.append(img4[:,:,1])
    img_g.append(img5[:,:,1])
    img_g=np.array(img_g)
    #print(img_g)

    img_r=[]
    img_r.append(img1[:,:,2])
    img_r.append(img2[:,:,2])
    img_r.append(img3[:,:,2])
    img_r.append(img4[:,:,2])
    img_r.append(img5[:,:,2])
    img_r=np.array(img_r)
    #print(img_r)

    new_b=single_hdr(img_b)
    print("finish b")
    new_g=single_hdr(img_g)
    print("finish g")
    new_r=single_hdr(img_r)
    print("finish r")


    # %% [markdown]
    # 

    # %%

    cv2.imwrite("b.hdr", new_b)
    cv2.imwrite("g.hdr", new_g)
    cv2.imwrite("r.hdr", new_r)

    new_b_reshape=np.expand_dims(new_b,2)
    new_g_reshape=np.expand_dims(new_g,2)
    new_r_reshape=np.expand_dims(new_r,2)

    hdr_image=np.concatenate((new_b_reshape,new_g_reshape,new_r_reshape),axis=2)
    print(hdr_image.shape)

    plt.figure(figsize=(12,8))
    plt.imshow(cv2.cvtColor(hdr_image.astype('float32'), cv2.COLOR_BGR2GRAY), cmap='jet')
    plt.colorbar()
    plt.savefig('radiance_map.png')
    print('done')
    #heatmap = cv2.applyColorMap(hdr_image.astype(np.uint8), cv2.COLORMAP_HOT)
    #cv2.imshow("heatmap", heatmap)
    #cv2.waitKey(0)
    plt.imshow(hdr_image, vmin=0, vmax=1)
    cv2.imwrite("output1.hdr", hdr_image)


    # %%
    #print(hdr_image)


    # %%



