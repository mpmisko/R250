import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

def make_video():
    folder = './videos/'
    imgs = []
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"): 
            img_file = os.path.join(folder, filename)
            files.append(img_file)

    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    files = sorted(files, key=numericalSort)
    for f in files:
        imgs.append(cv2.imread(f))

    height, width, _ = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter()
    opened = writer.open('video.mp4', fourcc, 10., (width, height), isColor=True)
    print(f"Writer opened: {opened}")
    print("Starting to make video ...")
     
    for img in tqdm(imgs):
        writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    writer.release()

if __name__ == "__main__":
    make_video()