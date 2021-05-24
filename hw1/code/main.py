import numpy as np
import cv2
import os
import glob
from hdr import HDR
from tonemap import Tonemap
import matplotlib.pyplot as plt
import argparse

class imageio():
    def __init__(self, root):
        self.root = root
        self.img_raw = []
        self.time = []
        self.g = None
        self.radiance_map = None
    def readraw(self):
        for img_path in sorted(glob.glob(os.path.join(self.root, "*.JPG")), key = lambda x : float(os.path.basename(x).split(".J")[0])):
            time = float(os.path.basename(img_path).split(".J")[0])
            self.time.append(time)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (576, 384))
            self.img_raw.append(img)
        self.img_raw = np.array(self.img_raw)
    def graph_g(self, g):
        self.g = g
        figure = plt.subplot()
        for func in self.g:
            plt.plot(func, range(len(func)))
        plt.ylabel("pixel value Z")
        plt.xlabel("log exposure X")
        plt.title("Response Curve")
        plt.savefig(os.path.join(self.root, "curve.png"))
        plt.close()
    def heatmap(self, radiance_map):
        plt.figure()
        plt.imshow(cv2.cvtColor(radiance_map.astype('float32'), cv2.COLOR_BGR2GRAY), cmap='jet')
        plt.colorbar()
        plt.savefig(os.path.join(self.root, 'radiance_map.png'))
        plt.title("Radiance Map")
        plt.close()
    def savefig(self, title, img):
        cv2.imwrite(os.path.join(self.root, title), img)
def parse_args():
    parser = argparse.ArgumentParser(description='VFX')
    parser.add_argument('--root', type=str, default='data/')
    parser.add_argument('--globaltone', action="store_true")
    parser.add_argument('--localtone', action="store_true")
    parser.add_argument('--key', type=float, default=0.6)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--sigma', type=float, default=15)
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    root = args.root
    io = imageio(root)
    io.readraw()
    hdr = HDR(io.img_raw, io.time)
    hdr.calculate()
    tonemap = Tonemap(hdr.radiance_map, args)
    if args.globaltone:
        io.savefig("globaltonemap.png", tonemap.globalmap())
    if args.localtone:
        io.savefig("localtonemap.png", tonemap.localmap())
    io.graph_g(hdr.g)
    io.heatmap(hdr.radiance_map)
