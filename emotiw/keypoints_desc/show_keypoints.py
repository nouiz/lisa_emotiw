from os import path
from os import listdir
from display_functions import displayImageWithKeypoints, displayImageWithBoundingbox, displayPoint, displayNum
from PIL import Image
import json
import scipy.io
import h5py
import tarfile
import sqlite3

def flatten(points):
    return [item for pair in points for item in pair] 

class DataReader(object):
    def __init__(self):
        self.images = []        
        self.points = []
        self.point_names = []

    def display_n(self, n, archive=None, drawFn=displayImageWithKeypoints, fnForDrawFn=displayPoint):
        """
        displays the first available n images
        from the data loaded at __init__ time,
        loading images from the given archive if
        provided.
        """
        
        action = lambda x: x

        if archive is not None:
            arch = tarfile.open(archive)
            action = lambda x: arch.extractfile(x)      

        i = 0
        while i < n and i < len(self.images):
            try:
                try:
                    img = Image.open(action(self.images[i]))
                    drawFn(img, self.points[i], fnForDrawFn)
                except IOError:
                    n = n+1
            finally:
                i = i+1

        if archive is not None:
            arch.close()

    def display_n_with_index(self, n, archive=None):        
        """
        displays point index alongside the points
        """

        self.display_n(n, archive, displayImageWithKeypoints, displayNum)

class InrialpesReader(DataReader):
    def __init__(self, top_dir='/data/lisa/data/faces/headpose/InrialpesHeadPose'):
        folders = [top_dir+'/Person'+str(i).zfill(2) for i in xrange(1, 15, 1)]
        self.images = []
        self.points = []
        self.point_names = ['face_x_center', 'face_y_center', 'face_width', 'face_height']

        for f in folders:
            files = listdir(f)
            
            for ff in files:
                if ff.endswith(".txt"):
                    fhandle = open(path.join(f, ff))
                    lines = fhandle.readlines()
                    self.images.append(path.join(f, lines[0].strip()))
                    lines = lines[3:] #skipping filename, empty line and "Face"
                    lst = [l.strip() for l in lines]
                    self.points.append(lst)

    def display_n(self, n):
        super(InrialpesReader, self).display_n(n, None, displayImageWithBoundingbox, None)

    def display_n_with_index(self, n, archive):
        pass
        #unsupported

class points_n_reader(DataReader):
    def __init__(self, top_dir, transform=lambda x: x):
        self.points = []

        for img in self.images:
            points = []
            f = open(path.join(top_dir, transform(img)))
            for l in f.readlines()[3:-1]:
                x_y = l.split(' ')
                points.extend([float(x_y[0]), float(x_y[1])])
            self.points.append(points)
            f.close()
    
class BioReader(points_n_reader):
    def __init__(self, top_dir='/data/lisa/data/faces/BioID/BioID-FaceDatabase-V1.2'):
        self.images = [top_dir+'/BioID_'+str(i).zfill(4)+'.pgm' for i in range(1520)]
        super(BioReader, self).__init__(path.join(path.dirname(top_dir), 'points_20/'), lambda x: x[-14:-3].lower() + 'pts')

class CaltechReader(DataReader):
    def __init__(self, archive='/data/lisa/data/faces/caltech/Caltech_WebFaces.tar', points_file='/data/lisa/data/faces/caltech/WebFaces_GroundThruth.txt'):
        self.archive = archive

        self.point_names = ['Leye-x', 'Leye-y', 'Reye-x', 'Reye-y', 'nose-x', 'nose-y', 'mouth-x', 'mouth-y']
        self.points = []
        self.images = []

        f = open(points_file)
        lines = f.readlines()

        for l in lines:
            words = l.strip().split(" ")
            self.images.append(words[0])

            point_lst = []
            for w in words[1:]:
                point_lst.append(w)
            
            self.points.append(point_lst)

        f.close()

    def display_n(self, n, drawFn=displayImageWithKeypoints, fnForDrawFn=displayPoint):
        return super(CaltechReader, self).display_n(n, self.archive, drawFn, fnForDrawFn)

    def display_n_with_index(self, n):
        return self.display_n(n, displayImageWithKeypoints, displayNum)
   
class WildPartsReader(DataReader):
    def __init__(self, fname='/data/lisa/data/faces/labeled_face_parts_in_the_wild/test_with_ids.csv', reldir='test'):
        f = open(fname)
        lines = f.readlines()
        f.close()
    
        point_entries = [x for x in [k.strip().split("\t") for k in lines] if x[2] == "average"]
        self.images = [path.join(path.dirname(fname), reldir, str(x[0])+".png") for x in point_entries]
        points = [x[3:] for x in point_entries]
        self.points = map(self.remove_meta, points)
        self.point_names = lines[0].strip().split("\t")[3:]
    
    def remove_meta(self, pts):
        lst = []
        for j in xrange(len(pts)):
            if pts[j] not in ('0', '1', '2') and not pts[j].endswith('\r\n'):
                lst.append(pts[j])
        return lst

class JsonBasedReader(DataReader):
    """
    Reader for the Json-based datasets.
    Expects images to be categorized by subject into subdirectories, with
    a different directory for point data and image data. File structure
    in one base directory should mirror filenames in the other, with point data 
    being in json format and images being in the format suggested by the 
    specified extension (which is case-sensitive and will be used to find 
    the image corresponding to a given point list).

    Images must be inside directories inside the top directories, and no deeper or
    shallower.
    """

    def __init__(self, top_dir='/data/lisa/data/faces/headpose/ncku', points_dir='/data/lisa/data/faces/headpose/mashapeKpts/ncku', ext='Jpg'):
        self.images = []
        self.points = []

        image_subjects = listdir(top_dir)
        points_subjects = listdir(points_dir)
        decoder = json.JSONDecoder()
        self.point_names = ['eye_left_x', 'eye_left_y', 'eye_right_x', 'eye_right_y',
                            'center_x', 'center_y', 'mouth_right_x', 'mouth_right_y',
                            'mouth_left_x', 'mouth_left_y', 'mouth_center_x', 'mouth_center_y',
                            'nose_x', 'nose_y']

        for folder in points_subjects:
            for f in listdir(path.join(points_dir, folder)):
                data = decoder.decode(open(path.join(points_dir, folder, f)).readlines()[0].strip())
                if len(data) > 0:
                    data = data[0]
                    
                    #The top-level construct is an array containing the keypoints object
                    
                    self.images.append(path.join(top_dir, folder, f[:-4]+ext))
                    self.points.append([data['eye_left']['x'], data['eye_left']['y'],
                                    data['eye_right']['x'], data['eye_right']['y'],
                                    data['center']['x'], data['center']['y'],
                                    data['mouth_right']['x'], data['mouth_right']['y'],
                                    data['mouth_left']['x'], data['mouth_left']['y'],
                                    data['mouth_center']['x'], data['mouth_center']['y'],
                                    data['nose']['x'], data['nose']['y']])

class NckuReader(JsonBasedReader):
    def __init__(self, top_dir='/data/lisa/data/faces/headpose/ncku', points_dir='/data/lisa/data/faces/headpose/mashapeKpts/ncku', ext='Jpg'):
        super(NckuReader, self).__init__(top_dir, points_dir, ext)
     
class HIIT6Reader(JsonBasedReader):
    def __init__(self, top_dir='/data/lisa/data/faces/headpose/HIIT6HeadPose/IIT6HeadPose/test', points_dir='/data/lisa/data/faces/headpose/mashapeKpts/HIIT6HeadPose/IIT6HeadPose/test', ext='png'):
        super(HIIT6Reader, self).__init__(top_dir, points_dir, ext)

class IHDPReader(JsonBasedReader):
    def __init__(self, top_dir='/data/lisa/data/faces/headpose/IHDPHeadPose', points_dir='/data/lisa/data/faces/headpose/mashapeKpts/IHDPHeadPose', ext='jpg'):
        super(IHDPReader, self).__init__(top_dir, points_dir, ext)

class InrialpesPoints(JsonBasedReader):
    def __init__(self, top_dir='/data/lisa/data/faces/headpose/InrialpesHeadPose', points_dir='/data/lisa/data/faces/headpose/mashapeKpts/InrialpesHeadPose', ext='jpg'):
        super(InrialpesPoints, self).__init__(top_dir, points_dir, ext)

class MultipieReader(DataReader):
    def __init__(self, image_dir='/data/lisa/data/faces/Multi-Pie/data', labels_dir='/data/lisa/data/faces/Multi-Pie/MPie_Labels/labels'):
        self.points = []
        self.images = []
        self.point_names = []
        
        #inspired by multipie.py (lisa_emotiw/emotiw/common/datasets/faces)
        folders = listdir(labels_dir)
        for folder in folders:
            files = listdir(path.join(labels_dir, folder))
    
            for f in files:
                self.points.append(flatten(
                        scipy.io.loadmat(
                            path.join(labels_dir, folder, f)
                        )['pts']))
                
                file_meaning = f.split('.')[0].split('_')
    
                session = 'session' + file_meaning[1]
                subject = file_meaning[0]
                identity = file_meaning[3][0:2] + '_' + file_meaning[3][2]            
    
                fname = path.join(image_dir, session, 'multiview', subject, 
                                    file_meaning[2], identity, '_'.join(file_meaning[0:5]) 
                                    + '.png')
                self.images.append(fname)
    
class AFWReader(DataReader):
    def __init__(self, top_dir='/data/lisa/data/faces/AFW/testimages'):
        self.points = []
        self.images = []
        self.point_names = []

        f = h5py.File(path.join(top_dir, 'anno.mat'), 'r')
        self.images = [path.join(top_dir, "".join(map(lambda x: chr(x), f[i].value))) 
                        for i in f['anno'].value[0]]
        #simply translate ints to chars for every char of the filename stored in the
        #first array of the dataset matrix

        self.points = [flatten(zipped) for zipped in  
                        #flatten the zipped values: 
                        #[(a, b), (c, d)] -> [a, b, c, d]
                    [zip(coords[0][0], coords[0][1]) for coords in     
                        #make pairs from (x,y) coords which are otherwise
                        #stored in two separate arrays
                         [map(lambda a: f[a].value, coord_ref) for coord_ref in
                            #extract the coords from the dataset
                            [flatten(f[coord_col]) for coord_col in f['anno'].value[3]]]]]

        f.close()
        
        #bbox = flatten(anno[1]) #format: x1 y1 x2 y2
        #yaw_pitch_roll = flatten(anno[2])
        
class AFLWReader(DataReader):
    def __init__(self, db='/data/lisa/data/faces/AFLW/aflw/data/aflw.sqlite', top_dir='/data/lisa/data/faces/AFLW/aflw/Images/aflw/data/flickr', fetch_ratio=0.1):

        #inspired by aflw.py (lisa_emotiw/emotiw/common/datasets/faces)
        #Quote from aforementioned script:
        #Note: subjects id starts from 39341 to 65384
        #End quote.

        self.points = []
        self.images = []
        self.point_names = []

        name_set = set()

        conn = sqlite3.connect(db)
        fetch_num = (65384-39341) * fetch_ratio
        
        for i in xrange(39341, int(39341 + fetch_num), 1):
            fileId = conn.execute('select file_id from Faces where face_id = ' + str(i)).fetchall()
            if len(fileId) <= 0: 
                continue

            result = conn.execute('select FeatureCoords.x, FeatureCoords.y, descr from ' +
                                   'FeatureCoords, FeatureCoordTypes where face_id = ' + 
                                    str(i) + 
                                    ' and FeatureCoordTypes.feature_id = FeatureCoords.feature_id ' +
                                    'order by FeatureCoords.feature_id').fetchall()

            self.points.append(flatten([(a[0], a[1]) for a in result]))
            name_set = name_set.union([a[2] for a in result])

            self.images.append(path.join(top_dir, fileId[0][0]))

        self.point_names = [name for name in name_set]

        conn.close()

 
