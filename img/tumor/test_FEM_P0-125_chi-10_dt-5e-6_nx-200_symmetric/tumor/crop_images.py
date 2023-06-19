import os
from PIL import Image

file_list = list(os.listdir('.'))
print(file_list)
file_list.sort()
for file in file_list:
    if not 'crop' in file:
        print(file)
        img = Image.open(file)
        width, height = img.size
        # Setting the points for cropped image
        left = width/12
        top = 0.08 * height
        right = 0.91 * width
        bottom = 0.92 * height
        im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
        iter = int(''.join(list(filter(str.isdigit, file))[0:]))
        print(iter)
        im_cropped.save(file.replace(f'-{iter}.png', f'_cropped-{int(iter/100)}.png'))
        os.remove(file)