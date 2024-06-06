import os
from PIL import Image

def check_images(s_dir, ext_list):
    bad_images = []
    bad_ext = []
    for root_dir, dirs, files in os.walk(s_dir):
        for file in files:
            if file.split('.')[-1].lower() not in ext_list:
                bad_ext.append(os.path.join(root_dir, file))
                print('Unacceptable file extension:', os.path.join(root_dir, file))
                continue
            try:
                img = Image.open(os.path.join(root_dir, file))
                img.verify()
            except (IOError, SyntaxError) as e:
                bad_images.append(os.path.join(root_dir, file))
                print('Bad file:', os.path.join(root_dir, file))  # print out the names of corrupt files
    return bad_images, bad_ext

s_dir = r'E:\resize'  # start directory
extensions = ['jpeg']  # acceptable extensions
bad_file_list, bad_ext_list = check_images(s_dir, extensions)

print("Corrupted image files:", bad_file_list)
print("Files with bad extensions:", bad_ext_list)
