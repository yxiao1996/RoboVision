import glob

neg_dir = '/home/yxiao1996/data/balls/neg/'
out_filename = './bg.txt'

def getItor(dir):
    itor = glob.iglob(dir+'*.jpeg')
    
    return itor

def openOut():
    out_file = open(out_filename, 'w')

    return out_file

def convert(filename, out_file):
    name = filename.split('/')[-2:]
    name_join = "/".join(name)
    out_file.write(name_join)
    out_file.write('\n')

itor = getItor(neg_dir)
out_file = openOut()

for filename in itor:
    convert(filename, out_file)

out_file.close()