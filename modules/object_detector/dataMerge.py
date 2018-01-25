import cv2
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom

dataset_dir = "/home/yxiao1996/data/balls/1-14/"
target_dir = "/home/yxiao1996/data/balls/"

def getItor(anno_dir):
    itor = glob.iglob(anno_dir+'*.xml')
    
    return itor

def getRoot(filename):
    in_file = open(filename)
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    return root



def saveXML(root, filename, indent="\t", newl="\n", encoding="utf-8"):
    rawText = ET.tostring(root)
    dom = minidom.parseString(rawText)
    with open(filename, 'w') as f:
        dom.writexml(f, "", indent, newl, encoding)


def moveImg(filename, offset):
    img = cv2.imread(str(dataset_dir + "pos/" + filename))
    img_num = int(filename.split(".")[0])
    
    new_img_num = img_num + offset
    new_filename = str(new_img_num) + ".jpeg"
    cv2.imwrite(str(target_dir + "pos/" + new_filename), img)

def getOffset():
    blob = glob.glob(target_dir + "pos/" + "*.jpeg")
    offset = len(blob)

    return offset

def moveAnno(root, offset):
    filename = root.find("filename").text
    img_num = int(filename.split(".")[0])
    new_img_num = img_num + offset
    new_img_filename = str(new_img_num) + ".jpeg"

    root.find("filename").text = new_img_filename
    root.find("path").text = target_dir + "pos" + new_img_filename
    new_filename = str(new_img_num) + ".xml"
    saveXML(root, str(target_dir + "Anno/" + new_filename))

offset = getOffset()
xml_itor = getItor(dataset_dir+"Anno/")
for xml_fn in xml_itor:
    root = getRoot(xml_fn)
    filename = root.find("filename").text
    moveImg(filename, offset)
    moveAnno(root, offset)