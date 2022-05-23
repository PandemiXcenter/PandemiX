# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:04:29 2020

@author: manno
"""
from PIL import Image
import glob
import os

#os.chdir(C:\Users\manno\OneDrive - Roskilde Universitet\Matematik fagmondul projekt\Kode\dwdxy)

# Create the frames
frames = []
imgs = sorted(glob.glob('*.png'), key=os.path.getmtime)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
    
#Save into a GIF file that loops forever
frames[0].save('png_to_gif.gif', format='GIF',
               append_images=frames[0:],
               save_all=True,
               duration=300, loop=0)

#os.chdir(C:\Users\manno\OneDrive - Roskilde Universitet\Matematik fagmondul projekt\Kode)