import shutil
import os
from progressbar import Percentage, ProgressBar,Bar,ETA

#filename = 'kitti_train.txt'

#with open(filename) as f:
#    content = f.readlines()
## you may also want to remove whitespace characters like `\n` at the end of each line
#content = [x.strip() for x in content]

#for i in pbar(content):
#    source_path = '/home/pskbalaji_project/project/dataset/kitti_raw_data/' + i
#    #print(source_path)
#    dest_path = './train/' + i[:i.rfind("/") + 1]
#    if not os.path.exists(dest_path):
#        os.makedirs(dest_path)rm -rf train
#    shutil.copy(source_path, dest_path)

filename = 'depth_val.txt'

with open(filename) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
final_content = []
pbar = ProgressBar(widgets=[Bar('>', '[', ']'), ' ', Percentage(), ' ', ETA()],maxval=len(content))
missing_file_count = 0
for i in pbar(content):
    source_path = '/home/pskbalaji_project/project/dataset/data_depth_annotated/train/' + i
    #print(source_path)
    if (os.path.isfile(source_path)):
        final_content.append(i)
        dest_path = './data/val/depth/' + i[:i.rfind("/") + 1]
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.copy(source_path, dest_path)
    elif (os.path.isfile('/home/pskbalaji_project/project/dataset/data_depth_annotated/val/' + i)):
        final_content.append(i)
        source_path = '/home/pskbalaji_project/project/dataset/data_depth_annotated/val/' + i        
        dest_path = './data/val/depth/' + i[:i.rfind("/") + 1]
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.copy(source_path, dest_path)
    else:
        missing_file_count += 1

print('Total number of files missing:', missing_file_count)
f = open("final_depth_val.txt", "w")
f.writelines("%s\n" % l for l in final_content)
f.close()
