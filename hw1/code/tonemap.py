import numpy as np
import cv2
from copy import deepcopy as copy

class Tonemap():
    def __init__(self, radiance_map, args):
        #parameters:
        self.key = args.key
        self.bottom = 1e-9
        self.threshold = args.threshold
        self.sigma = args.sigma
        #maps:
        self.L_w = radiance_map
        self.L_w_bar = np.exp(np.mean(np.log(self.L_w + self.bottom), axis=(0, 1)))
        self.L_m = np.divide(self.key * self.L_w, self.L_w_bar)
        self.L_white = np.sort(self.L_m.reshape(-1))
        self.L_white = self.L_white[int(0.9 * len(self.L_white))]
    def globalmap(self):
        L_d = np.divide(self.L_m * (1 + (self.L_m / (self.L_white**2))), 1 + self.L_m)
        L_d = np.where(L_d < 1, L_d, 1)
        L_d = np.where(L_d < 0, 0, L_d)
        L_d = L_d * 255 
        return L_d
    def localmap(self):
        s = 3
        L_s = cv2.GaussianBlur(self.L_m, (s, s), 0)
        L_blur = copy(L_s) 
        done = np.ones(self.L_w.shape)
        while True:
            L_s = cv2.GaussianBlur(self.L_m, (s, s), 0)
            L_s_plus = cv2.GaussianBlur(self.L_m, (s+2, s+2), 0)
            norm = (2**self.sigma * self.key) / (s**2) + L_s 
            V_s = np.divide((L_s - L_s_plus), norm)
            L_blur = np.where(np.logical_and(abs(V_s) < self.threshold, done), L_s_plus, L_blur)
            done = np.where(abs(V_s) < self.threshold, 0, done)
            if not done.any():
                break
            s += 2
        L_d = np.divide(self.L_m * (1 + (self.L_m / (self.L_white**2))), 1 + L_blur)
        L_d = np.where(L_d < 1, L_d, 1)
        L_d = np.where(L_d < 0, 0, L_d)
        L_d = L_d * 255 
        return L_d
        
