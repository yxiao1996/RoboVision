import cv2
import glob

source_dir = '/home/yxiao1996/data/balls/1-14/pos_raw/'
target_dir = '/home/yxiao1996/data/balls/1-14/pos/'
img_format = '.jpeg'

def getImgNum():
    blob = glob.glob(target_dir+'*'+img_format)
    num_img = len(blob)

    return num_img

def procImage(img_dir, save_dir, img_num, size=(960,720)):
    print "processing images..."
    img_iter = glob.iglob(img_dir+'*')
    img_names = []
    imgs = []
    for i, img in enumerate(img_iter):
        img_names.append(img)
        image = cv2.imread(img)
        img_resize = cv2.resize(image, size)
        save_name = save_dir + str(i+img_num) + img_format
        cv2.imwrite(save_name, img_resize)

if __name__ == '__main__':
    img_num = getImgNum()
    procImage(source_dir, target_dir, img_num)