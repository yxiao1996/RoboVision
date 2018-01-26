import glob
import xml.etree.ElementTree as ET

anno_dir = '/home/yxiao1996/data/balls/Anno/'
out_filename = './info.dat'

def getItor(anno_dir):
    itor = glob.iglob(anno_dir+'*.xml')
    
    return itor

def getRoot(filename):
    in_file = open(filename)
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    return root

def openOut():
    out_file = open(out_filename, 'w')

    return out_file

def convert(root, out_file):
    size = root.find('size')
    path = root.find('path').text
    name = path.split('/')[-2:]
    name_join = "/".join(name)
    # write image name
    out_file.write(name_join)
    out_file.write(' ')
    # write number of objects
    num_obj = len(root.findall('object'))
    out_file.write(str(num_obj))
    out_file.write(' ')
    # write bounding box
    bun_box = root.find('object').find('bndbox')
    max_x = float(bun_box.find('xmax').text)
    min_x = float(bun_box.find('xmin').text)
    max_y = float(bun_box.find('ymax').text)
    min_y = float(bun_box.find('ymin').text)
    #x = int((max_x + min_x) / 2)
    #y = int((max_y + min_y) / 2)
    #width = int(max_x - min_x)
    #height = int(max_y - min_y)
    x = int(min_x)
    y = int(min_y)
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    out_file.write(str(x))
    out_file.write(' ')
    out_file.write(str(y))
    out_file.write(' ')
    out_file.write(str(width))
    out_file.write(' ')
    out_file.write(str(height))
    out_file.write(' ')
    out_file.write('\n')

#def pipeline():

out_file = openOut()
filename_itor = getItor(anno_dir)
for filename in filename_itor:
    root = getRoot(filename)
    convert(root, out_file)
out_file.close()