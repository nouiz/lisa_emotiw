import os
import Image

path = '/data/lisa/data/faces/Multi-Pie/data/session{}/multiview/'

sessions = ['01', '02', '03', '04']

for i in sessions:
    print 'session', i
    for root, subdir, files in os.walk(path.format(i)):
        print root
        if '19_1' in root:
            for file in files:
                if os.path.splitext(file)[-1] in ['.png']:
                    filePath = os.path.join(root, file)
                    img = Image.open(filePath)
                    print filePath
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    #img.save(filePath) 
            

