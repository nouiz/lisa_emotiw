import glob
import numpy
import cv2
from PIL import Image, ImageDraw
from subprocess import call
import shutil
#import ipdb
import pickle
import sys
import os


color_list = ['#CD0000', '#1E90FF', '#FFFF00', '#00EE00', '#FF34B3',
                '#63B8FF', '#FFF68F', '#8E8E38', '#00C78C', '#ff00ff', '#00ffff',
                '#ffff00', '#5528b2', '#3AEAA4', '#E4BFBA', '#3F9197', '#F83D17',
                '#30577B', '#5E7B30', '#5635E7', '#8575CD', '#3DC72D', '#C72D7F',
                '#3D0C20', '#1B31BF', '#20BF1B', '#133B12', '#3450AD', '#57689E',
                '#3B315E', '#F7CE52', '#CAC8DB', '#393654', '#080717', '#0AD1D1',
                '#7F3FC4', '#C43F93', '#A8B1E6', '#151C42', '#84A383', '#0FBD06']
color_list = color_list * 10

def get_clip_lists(path, emo):
    list = []
    for e in emo:
        tmp_list = glob.glob("{}{}*".format(path, e.lower()))
        tmp_list.sort()
        list.extend(tmp_list)
        
    rval = []
    for item in list:

        rval.append(item.split('/')[-1])
    return rval

def get_frames(path, clips):

    rval = {}
    for clip in clips:
        list = glob.glob("{}{}/*png".format(path, clip))
        rval[clip] = list

    return rval

def get_bounding_boxes(path, clips):

    rval = {}
    for clip in clips:
        list = glob.glob("{}{}/*.txt".format(path, clip))
        #print 'list', "{}{}/*.txt".format(path, clip)
        #list = [item.split('/')[-1] for item in list]
        clip_holder = {}
        for item in list:
            name = int(item.split('/')[-1].rstrip('.txt').split('-')[-1])
            val = numpy.loadtxt(item, delimiter=',')
            if val.ndim == 1:
                val = val.reshape((1,4))
            clip_holder[name] = val.tolist()
        rval[clip] = clip_holder
    return rval

def isclose(a, b, thre = 10):

    if numpy.linalg.norm(a-b) < thre:
        return True
    else:
        return False

def get_distance(a, b):

    center_a = numpy.array([a[2] - a[0], a[3] - a[1]])
    center_b = numpy.array([b[2] - b[0], b[3] - b[1]])
    return numpy.linalg.norm(center_a - center_b)

def get_rel_distance(a, b):

    def size(x):
        return float(x[2] - x[0]) * (x[3] - x[1])

    center_a = numpy.array([a[2] - a[0], a[3] - a[1]])
    center_b = numpy.array([b[2] - b[0], b[3] - b[1]])
    return numpy.linalg.norm(center_a - center_b) / numpy.mean([size(a), size(b)])



def is_center_close(a, b, thre = 10):

    if get_distance(a, b) < thre:
        return True
    else:
        return False

def is_same_size(a, b, thre = 10):
    def size(x):
        return float(x[2] - x[0]) * (x[3] - x[1])

    #diff = numpy.abs(size(a) - size(b))
    #print diff
    #if numpy.abs(size(a) - size(b)) < thre:
    size_a = size(a)
    size_b = size(b)
    if max(size_a, size_b) / min(size_a, size_b) < thre:
        return True
    else:
        return False

def check_loc(data, loc_thre, size_thre):

    failed = {}
    for clip in data.keys():
        frames = data[clip].keys()
        frames.sort()
        i = 0
        failed[clip] = []
        for frame in frames:
            if i == 0:
                prev = data[clip][frame]
                i = 1
                continue
            save = False
            if not is_center_close(data[clip][frame], prev, loc_thre):
                #print 'loc ', clip, frame, prev, data[clip][frame]
                #failed[clip].append(frame)
                if not is_same_size(data[clip][frame], prev, size_thre):
                    print 'size ', clip, frame, prev, data[clip][frame]
                    failed[clip].append(frame)
            prev = data[clip][frame]
    return failed

def get_previous_boxes(curr, prev, boxes, similar_thr):

    prev = Image.open(prev)
    curr = Image.open(curr)

    rval = []
    for box in boxes:
        box = [int(item) for item in box]

        prev = prev.crop(box)
        rvals = []
        pads = []
        for x in range(-21, 22, 3):
            for y in (-21, 22, 3):
                pads.append([x, y])
        for pad in pads:
            tmp_box = [box[0] + pad[0], box[1] + pad[1], box[2] + pad[0], box[3] + pad[1]]
            for item in tmp_box:
                if item < 0:
                    continue
            curr_tmp = curr.crop(tmp_box)
            rvals.append(similarness(prev, curr))

        if max(rvals) > similar_thr: # 0.10:
            pad = pads[numpy.argmax(rvals)]
            rval.append([box[0] + pad[0], box[1] + pad[1], box[2] + pad[0], box[3] + pad[1]])

    return rval

def overlap(a, b):

    def size(x):
        return float(x[2] - x[0]) * (x[3] - x[1])

    x11 = a[0]
    y11 = a[1]
    x12 = a[2]
    y12 = a[3]
    x21 = b[0]
    y21 = b[1]
    x22 = b[2]
    y22 = b[3]
    x_overlap = numpy.max([0, numpy.min([x12,x22]) - numpy.max([x11,x21])])
    y_overlap = numpy.max([0, numpy.min([y12,y22]) - numpy.max([y11,y21])])
    return (x_overlap * y_overlap ) / numpy.mean([size(a), size(b)])

def similarness(i1,i2):
    """
    Return the correlation distance be1tween the histograms. This is 'normalized' so that
    1 is a perfect match while -1 is a complete mismatch and 0 is no match.
    """
    ## Open and resize images to 200x200
    #i1 = Image.open(image1).resize((200,200))
    #i2 = Image.open(image2).resize((200,200))

    # Get histogram and seperate into RGB channels
    i1hist = numpy.array(i1.histogram()).astype('float32')
    i1r, i1b, i1g = i1hist[0:256], i1hist[256:256*2], i1hist[256*2:]
    # Re bin the histogram from 256 bins to 48 for each channel
    i1rh = numpy.array([sum(i1r[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i1bh = numpy.array([sum(i1b[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i1gh = numpy.array([sum(i1g[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    # Combine all the channels back into one array
    i1histbin = numpy.ravel([i1rh, i1bh, i1gh]).astype('float32')

    # Same steps for the second image
    i2hist = numpy.array(i2.histogram()).astype('float32')
    i2r, i2b, i2g = i2hist[0:256], i2hist[256:256*2], i2hist[256*2:]
    i2rh = numpy.array([sum(i2r[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i2bh = numpy.array([sum(i2b[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i2gh = numpy.array([sum(i2g[i*16:16*(i+1)]) for i in range(16)]).astype('float32')
    i2histbin = numpy.ravel([i2rh, i2bh, i2gh]).astype('float32')

    return cv2.compareHist(i1histbin, i2histbin, 0)

def assign_tube(tubes, boxes, frame, distance_thr, size_thr, overlap_thr):

    # find the latest frame in each facetube
    prev_boxes = []
    for item in tubes:
        frames = item.keys()
        frames.sort()
        prev_boxes.append(item[frames[-1]])

    distances = []
    dist_order = []
    # find all distances between current frame bounding boxes and exsiting facetubes
    for i, box in enumerate(boxes):
        for j, prev_box in enumerate(prev_boxes):
            distances.append(get_rel_distance(box, prev_box))
            dist_order.append([i, j])

    # sort distances
    ind = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])]
    # assign bounding boxes to face tubes in order of distances
    box_list = list(range(len(boxes)))

    #print distances
    for i in ind:
        box_ind = dist_order[i][0]
        tube_ind = dist_order[i][1]

        # check relative distance
        if distances[i] > distance_thr:
            #print 'dist'
            continue

        # check size
        last_tube_frame = numpy.max(tubes[tube_ind].keys())
        if not is_same_size(tubes[tube_ind][last_tube_frame], boxes[box_ind], size_thr):
            #print 'size'
            continue

        #print overlap(tubes[tube_ind][last_tube_frame], boxes[box_ind])
        if overlap(tubes[tube_ind][last_tube_frame], boxes[box_ind]) < overlap_thr:
            #print 'overlap'
            continue

        if box_ind in box_list:
            box_ind = int(box_ind)
            #ipdb.set_trace()
            # check if faces look similar
            #if similar_faces(boxes[box_ind], prev_boxes[tube_ind], ):
            tubes[int(tube_ind)][frame] = boxes[box_ind]
            box_list.remove(box_ind)

    # make new face tubes for remaining boxes
    for i in box_list:
        tubes.append({frame : boxes[i]})

    return tubes

def get_face_tubes(frames, boxes, failed, img_path, tube_path, tube_full_path, distance_thr, size_thr, overlap_thr, similar_thr):
    
    def find_previous(data, ind):
        keys = data.keys()
        keys.sort(reverse = True)

        for i in keys:
            if ind > i:
                return i
    def find_range(data):
        ids = [int(item.split('-')[-1].rstrip('.png')) for item in data]
        return range(min(ids), max(ids))

    face_tubes = {}
    for clip in frames.keys():
        if len(boxes[clip]) == 0:
            continue
        print "Finding face-tubes for clip: {}".format(clip)
        color_ind = 0
        frame_range = find_range(frames[clip])
        box_range = [min(boxes[clip].keys()), max(boxes[clip].keys())]
        face_tubes[clip] = []
        for frame in frame_range:
            img = "%s%s-%03d.png" % (img_path, clip, frame)
            # if we have bounding box
            if frame in boxes[clip]:
                face_tubes[clip] = assign_tube(face_tubes[clip], boxes[clip][frame], frame, distance_thr, size_thr, overlap_thr)
            # else look for previous frames for linear interpolation
            else:
                prev = find_previous(boxes[clip], frame)
                
                if prev is not None:
                    
                    prev_img = "%s%s-%03d.png" % (img_path, clip, prev)
                    prev_boxes = get_previous_boxes(img, prev_img, boxes[clip][prev], similar_thr)
                    
                    if len(prev_boxes) > 0:
                        face_tubes[clip] = assign_tube(face_tubes[clip], prev_boxes, frame, distance_thr, size_thr, overlap_thr)
    return face_tubes

def save_tube(tubes, src_path, des_path):

    for clip in tubes.keys():
        print "Saving face tube images for clip: {}".format(clip)
        for i, tube in enumerate(tubes[clip]):
            frames = tube.keys()
            frames.sort()
            for frame in frames:
                save_name = "%s%s-%d-%03d.jpg" % (des_path, clip, i, frame)
                orig_name = "%s%s-%03d.png" % (src_path, clip, frame)
                save_crop(orig_name, save_name, tube[frame])

def save_tube_on_full_image(frames, tubes, src_path, des_path):

    def get_boxes(tubes, frame):
        boxes = []
        colors = []
        for tube in tubes:
            if frame in tube.keys():
                boxes.append(tube[frame])
                colors.append(tubes.index(tube))
        return boxes, colors

    for clip in frames.keys():

        print "Saving face tube on original images for clip: {}".format(clip)
        if not tubes.has_key(clip):
            print "There is no data for this clip"
            continue
        
        for frame in frames[clip]:
            frame = int(frame.split('-')[-1].rstrip('.png'))
            save_name = "%s%s-%03d.jpg" % (des_path, clip, frame)
            orig_name = "%s%s-%03d.png" % (src_path, clip, frame)
            boxes, colors = get_boxes(tubes[clip], frame)
            if len(boxes) > 0:
                draw(orig_name, save_name, boxes, [color_list[i] for i in colors])
            else:
                shutil.copyfile(orig_name, save_name)

def save_crop(img, facetube_path, box):
    box = [int(item) for item in box]
    img = Image.open(img)
    img = img.crop(box)
    img.save(facetube_path)

def draw(img, facetube_path, boxes, colors):
    img = Image.open(img)
    draw = ImageDraw.Draw(img)
    for box, color in zip(boxes, colors):
        draw.line((box[0], box[1], box[0], box[3]), fill = color, width=5)
        draw.line((box[0], box[1], box[2], box[1]), fill = color, width=5)
        draw.line((box[2], box[1], box[2], box[3]), fill = color, width=5)
        draw.line((box[2], box[3], box[0], box[3]), fill = color, width=5)
    img.save(facetube_path)

def main():

    #clips_path = '/data/lisa/data/faces/EmotiW/AFEW_2_Distribute/'
    #clips_path =  "/data/lisatmp2/EmotiWTest/Test_Vid_Distr/Data/"
    
    clips_path = "/data/lisatmp2/emo_video/new_clips/BoundingBox/"
    #clips_path = "/Users/zhenzhou/Dropbox/CodingProjects/bx_path/"
    
    #frame_path = "/data/lisa/data/faces/EmotiW/images/"
    #frame_path =  "/data/lisatmp2/EmotiWTest/Test_Vid_Distr/ExtractedFrame/"
    #frame_path = "/Users/zhenzhou/Dropbox/CodingProjects/ExtractedFrames/"
    frame_path = "/data/lisatmp2/emo_video/new_clips/ExtractedFrames/"
    
    #bx_path = '/data/lisa/data/faces/EmotiW/picasa_boxes/'
    #bx_path =  "/Users/zhenzhou/Dropbox/CodingProjects/bx_path/"
    bx_path = "/data/lisatmp2/emo_video/new_clips/BoundingBox/"
    
    #facetube_path = '/data/lisa/data/faces/EmotiW/picasa_face_tubes/v2/'
    facetube_path =  "/data/lisatmp2/emo_video/new_clips/FaceTube/"
    
    if not os.path.exists(facetube_path):
        os.mkdir(facetube_path)
    
    
    #bx_on_frame_path = "/data/lisa/data/faces/EmotiW/picasa_face_tubes_full_images/v2/"
    bx_on_frame_path =  "/data/lisatmp2/emo_video/new_clips/bbox_on_image/"
    if not os.path.exists(bx_on_frame_path):
        os.mkdir(bx_on_frame_path)
        
    #tube_pickle_path = "/data/lisa/data/faces/EmotiW/picasa_tubes_pickles/"
    tube_pickle_path =  "/data/lisatmp2/emo_video/new_clips/pickle/"
    if not os.path.exists(tube_pickle_path):
        os.mkdir(tube_pickle_path)
        
    report_path = "/data/lisatmp2/emo_video/new_clips/Report/"
    if not os.path.exists(report_path):
        os.mkdir(report_path)

    size_thre = 10
    #loc_thre = 200
    distance_thr = .008
    overlap_thr = 0.08
    similar_thr = 0.97

    stats = {}    
    lstat = {}
    
    emots = ["happy", "angry", "sad", "fear", "suprise"]
    

    report = open(report_path+'facetube_report', 'w')
    #clips = get_clip_lists("/data/lisatmp2/emo_video/new_clips/emotion_dataset/happy")
    
    for emo in emots:
    
		clips = get_clip_lists(clips_path, emo)
  
		for clip_name in clips:
			print "clip:", clip_name
			clip = []
			clip.append(clip_name)
		
			frames = get_frames(frame_path, clip) #path of all the frames in the clip name
	
			bx = get_bounding_boxes(bx_path, clip)

			#failed = check_loc(bx, loc_thre, size_thre)
		
			ft_emo_path = "{}{}/".format(facetube_path, clip_name)
			bx_emo_path = "{}{}/".format(bx_path, clip_name)
			frame_emo_path = "{}{}/".format(frame_path, clip_name)
			bx_on_frame = "{}{}/".format(bx_on_frame_path, clip_name)
		
			if not os.path.exists(ft_emo_path):
				call(['mkdir', ft_emo_path])
			
			if not os.path.exists(bx_on_frame):
				call(['mkdir', bx_on_frame])
		
			failed = []
			try:
				print 'frames ', frames
				print 'bx ', bx
				print 'frame_emo_path ', frame_emo_path
				print 'ft_emo_path ', ft_emo_path
				print 'bx_emo_path ', bx_emo_path
				face_tubes = get_face_tubes(frames, bx, failed, frame_emo_path, \
							ft_emo_path, bx_emo_path, distance_thr, size_thre, overlap_thr, similar_thr)
				save_tube(face_tubes,  frame_emo_path, ft_emo_path)
	
				save_tube_on_full_image(frames, face_tubes,  frame_emo_path, bx_on_frame)
				print clip_name, ": done!"
				report.write(clip_name + ": done")
			
				output_name = "{}{}.pkl".format(tube_pickle_path, clip_name)
				with open(output_name, 'w') as out_f:
					pickle.dump(face_tubes, out_f)
			except:
				print clip_name, ": failed!"
				report.write(clip_name + ": failed")
		
		
def compute_facetube(clip, frame_list, bbox_list, ft_dir, bb_on_image_dir):
    '''
    clip : the avi video path
    frame_list: list containing the frames for the clip
    bbox_list: list containing the bbox coordinate for the list
    ft_dir: directory for putting the facetubes
    bb_on_image_dir: directory for putting the bounding-box-on-image images
    '''
    
if __name__ == "__main__":
    main()
    #dev_main()
