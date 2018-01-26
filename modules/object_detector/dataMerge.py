import cv2
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom

dataset_dir = "/home/yxiao1996/data/balls/1-25/"
target_dir = "/home/yxiao1996/data/balls/"

def getItor(anno_dir):
    itor = glob.iglob(anno_dir+'*.xml')
    
    return itor

def getNegItor():
    itor = glob.iglob(dataset_dir+"neg/"+"*.jpeg")

    return itor

def getRoot(filename):
    in_file = open(filename)
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    return root

def saveXML(root, filename, indent="\t", newl="", encoding="utf-8"):
    rawText = ET.tostring(root)
    dom = minidom.parseString(rawText)
    with open(filename, 'w') as f:
        dom.writexml(f, "", indent, newl, encoding)


def moveImg(filename, offset, pos=True):
    if pos:
        img = cv2.imread(str(dataset_dir + "pos/" + filename))
    else:
        img = cv2.imread(str(dataset_dir + "neg/" + filename))
    img_num = int(filename.split(".")[0])
    
    new_img_num = img_num + offset
    new_filename = str(new_img_num) + ".jpeg"
    if pos:
        cv2.imwrite(str(target_dir + "pos/" + new_filename), img)
    else:
        cv2.imwrite(str(target_dir + "neg/" + new_filename), img)

def getOffset(pos=True):
    if pos:
        blob = glob.glob(target_dir + "pos/" + "*.jpeg")
    else:
        blob = glob.glob(target_dir + "neg/" + "*.jpeg")
    offset = len(blob)

    return offset

def moveAnno(root, offset):
    filename = root.find("filename").text
    img_num = int(filename.split(".")[0])
    new_img_num = img_num + offset
    new_img_filename = str(new_img_num) + ".jpeg"

    root.find("filename").text = new_img_filename
    root.find("path").text = target_dir + "pos/" + new_img_filename
    new_filename = str(new_img_num) + ".xml"
    saveXML(root, str(target_dir + "Anno/" + new_filename))

offset = getOffset()
print ("pos offset", offset)
xml_itor = getItor(dataset_dir+"Anno/")
for xml_fn in xml_itor:
    root = getRoot(xml_fn)
    filename = root.find("filename").text
    moveImg(filename, offset)
    moveAnno(root, offset)

neg_offset = getOffset(pos=False)
print ("neg offset", neg_offset)
neg_itor = getNegItor()
for path in neg_itor:
    filename = path.split('/')[-1]
    moveImg(filename, neg_offset, pos=False)