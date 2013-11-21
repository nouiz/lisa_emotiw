import glob
import numpy
import cv2
from PIL import Image, ImageDraw
import shutil
import pickle
import os
import scipy.io


color_list = ['#CD0000', '#1E90FF', '#FFFF00', '#00EE00', '#FF34B3',
                '#63B8FF', '#FFF68F', '#8E8E38', '#00C78C', '#ff00ff', '#00ffff',
                '#ffff00', '#5528b2', '#3AEAA4', '#E4BFBA', '#3F9197', '#F83D17',
                '#30577B', '#5E7B30', '#5635E7', '#8575CD', '#3DC72D', '#C72D7F',
                '#3D0C20', '#1B31BF', '#20BF1B', '#133B12', '#3450AD', '#57689E',
                '#3B315E', '#F7CE52', '#CAC8DB', '#393654', '#080717', '#0AD1D1',
                '#7F3FC4', '#C43F93', '#A8B1E6', '#151C42', '#84A383', '#0FBD06']
color_list = color_list * 10

def get_clip_lists(path):

    list = glob.glob("{}*.avi".format(path))
    rval = []
    for item in list:
        rval.append(item.split('/')[-1].rstrip('.avi'))
    return rval

def get_frames(path, clips):

    rval = {}
    for clip in clips:
        list = glob.glob("{}{}*png".format(path, clip))
        rval[clip] = list

    return rval

def get_bounding_boxes(path, clips):

    rval = {}
    for clip in clips:
        list = glob.glob("{}{}*.txt".format(path, clip))
        #list = [item.split('/')[-1] for item in list]
        clip_holder = {}
        for item in list:
            name = int(item.split('/')[-1].rstrip('.txt').split('-')[-1].split('_')[0])
            val = numpy.loadtxt(item, delimiter=',')
            if val.ndim == 1:
                try:
                    val = val.reshape((1,4))
                except ValueError:
		    import pdb; pdb.set_trace()
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

def get_face_tubes(frames, boxes, failed, img_path, tube_path, distance_thr, size_thr, overlap_thr, similar_thr):

    def find_previous(data, ind):
        keys = data.keys()
        keys.sort(reverse = True)

        for i in keys:
            if ind > i:
                return i

    def find_range(data):
        ids = [int(item.split('-')[-1].rstrip('.png').split('_')[0]) for item in data]
        return range(min(ids), max(ids)+1)

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
            img = "%s%s-%03d_.png" % (img_path, clip, frame) 

            # if we have bounding box
            if frame in boxes[clip]:
                face_tubes[clip] = assign_tube(face_tubes[clip], boxes[clip][frame], frame, distance_thr, size_thr, overlap_thr)

            # else look for previous frames for linear interpolation
            else:
                prev = find_previous(boxes[clip], frame)
                if prev is not None:
                    prev_img = "%s%s-%03d_.png" % (img_path, clip, prev)
                    prev_boxes = get_previous_boxes(img, prev_img, boxes[clip][prev], similar_thr)
                    if len(prev_boxes) > 0:
                        face_tubes[clip] = assign_tube(face_tubes[clip], prev_boxes, frame, distance_thr, size_thr, overlap_thr)

    return face_tubes

def smooth(x,window_len=11):
        rval = []
        for idx, value in enumerate(x):
            interval = []
            if idx - window_len/2 < 0:
                lefthand_side = x[: idx]
            else:
                lefthand_side = x[idx - window_len/2: idx]
            righthand_side = x[idx+1:idx+1+window_len/2]

            res1 = numpy.abs((lefthand_side.mean() - x[idx]) / lefthand_side.mean())
            res2 = numpy.abs((righthand_side.mean() - x[idx]) / righthand_side.mean())
            temp_interval = numpy.hstack((lefthand_side, righthand_side))
            res3 = numpy.abs((temp_interval.mean() - x[idx]) / temp_interval.mean())

            if res1 > 0.10 and res2 < 0.10:
                interval = numpy.hstack((x[idx], righthand_side))
            elif  res1 < 0.10 and res2 > 0.10:
                interval = numpy.hstack((lefthand_side, x[idx]))
            else:
                interval = numpy.hstack((lefthand_side, x[idx], righthand_side))
            moving_average = numpy.mean(interval)
            if ( abs(moving_average - x[idx]) / moving_average ) > 0.09:
                if len(rval):
                    moving_average = x[idx]
                else:
                    moving_average = x[idx]
            rval.append(moving_average)
        return numpy.array(rval)

def save_tube(tubes, src_path, des_path, size, what_to_save):
    # First smoothing window size where we smooth opposite corners coordinates.
    opposite_corners_window_size = 11
    # Second smoothing window size where we smooth the center of the bounding box.
    bb_center_window_size = 11
    deg = 0
    deg2 = 1
    threshold_smallest_bb_area = 40*40
    #threshold_similar = 0.7
    threshold_similar = 0.52

    if not os.path.isdir(des_path) and what_to_save != 'bbox_coords':
        os.makedirs(des_path)

    if True:
        facetubes = {}
        for clip in tubes.keys():
            facetubes[clip] = [dict(zip(tube.keys(), [[] for i in xrange(len(tube.keys()))])) for clip_i, tube in enumerate(tubes[clip])]
            print "Saving face tube images for clip: {}".format(clip)
            for clip_i, tube in enumerate(tubes[clip]):
                frames = tube.keys()
                frames.sort()
                # Compute the bounding box moving average on the opposite corners only.
                # Compute the moving average along each coordinates of the opposite corners.
                smoothed_bb_coords = []
                all_bb_coords = []
                all_sq_bb_center_coords = []
                all_sq_bb_length = []
                tube_arr = numpy.array([tube[f] for f in frames])

                for i in xrange(4):
                    smoothed_coord = smooth(tube_arr[:, i], opposite_corners_window_size)
                    smoothed_coord = smoothed_coord.reshape((len(smoothed_coord), 1))
                    if len(smoothed_bb_coords):
                        smoothed_bb_coords = numpy.hstack((smoothed_bb_coords, smoothed_coord))
                    else:
                        smoothed_bb_coords = smoothed_coord
                # NOTE: ROUND
                smoothed_bb_coords = numpy.round(smoothed_bb_coords)

                for idx, frame in enumerate(frames):
                    sq_bb_coords, sq_bb_center_coords, sq_bb_length = smoothing(smoothed_bb_coords[idx])
                    all_bb_coords.append(sq_bb_coords)
                    all_sq_bb_center_coords.append(sq_bb_center_coords)
                    all_sq_bb_length.append(sq_bb_length)

                all_sq_bb_center_coords = numpy.array(all_sq_bb_center_coords)
                # Compute the moving average on the center point.
                smoothed_sq_bb_center_coords = []
                for i in xrange(all_sq_bb_center_coords.shape[1]):
                    smoothed_coord = smooth(all_sq_bb_center_coords[:,i], bb_center_window_size)
                    smoothed_coord = smoothed_coord.reshape((len(smoothed_coord), 1))
                    if len(smoothed_sq_bb_center_coords):
                        smoothed_sq_bb_center_coords = numpy.hstack((smoothed_sq_bb_center_coords, smoothed_coord))
                    else:
                        smoothed_sq_bb_center_coords = smoothed_coord

                smoothed_sq_bb_center_coords = numpy.array(smoothed_sq_bb_center_coords)
                # NOTE: ROUND
                smoothed_sq_bb_center_coords = numpy.round(smoothed_sq_bb_center_coords)

                from numpy import polyfit, poly1d

                xdata1 = []
                xdata2 = []
                ydata1 = []
                ydata2 = []

                xdata = numpy.arange(0, len(all_sq_bb_length))
                ydata = numpy.array(all_sq_bb_length)
                pars = polyfit(xdata, ydata, deg)
                poly = poly1d(pars)

                pars2 = polyfit(xdata, ydata, deg2)
                poly2 = poly1d(pars2)

                smoothed_sq_bb_length = []
                for i in xrange(len(all_sq_bb_length)):
                    if all_sq_bb_length[i] / poly(i) > 1.5:
                        smoothed_sq_bb_length.append(poly2(i))
                    else:
                        smoothed_sq_bb_length.append(poly(i))

                smoothed_sq_bb_length = numpy.round(numpy.array(smoothed_sq_bb_length))

                for idx in xrange(len(smoothed_sq_bb_center_coords)):
                    frame = frames[idx]
                    orig_name = "%s%s-%03d_.png" % (src_path, clip, frame)
                    img = cv2.imread(orig_name)
                    if img is None:
                        continue

                    center_x, center_y = smoothed_sq_bb_center_coords[idx]
                    length = smoothed_sq_bb_length[idx]
                    # Get top left and bottom right corners coordinates based on the coordinate
                    # of the middle point.
                    pt1_x, pt1_y = int(center_x - (length/2.)), int(center_y - (length/2.))
                    pt2_x, pt2_y = int(center_x + (length/2.)), int(center_y + (length/2.))

                    pt1_x, pt2_x = numpy.clip([pt1_x, pt2_x], 0, img.shape[1]-1)
                    pt1_y, pt2_y = numpy.clip([pt1_y, pt2_y], 0, img.shape[0]-1)

                    # Check if there is a black border around the picture. We will trim
                    # this black border if it is the case.
                    # Get the min and max values x and y can take without the black border.
                    pts_y = numpy.arange(0, img.shape[0])
                    pts_x = numpy.arange(0, img.shape[1])
                    min_y = 0
                    max_y = img.shape[0] - 1
                    min_x = 0
                    max_x = img.shape[1] - 1
                    for indx, y in enumerate(pts_y):
                        if img[y,0:img.shape[1]].mean() <= 1:
                            min_y = pts_y[indx+1]
                            assert indx < len(pts_y)
                        else:
                            break
                    for indx, y in enumerate(pts_y):
                        if img[pts_y[-indx-1],0:img.shape[1]].mean() <= 1:
                            if indx == len(pts_y):
                                break
                            max_y = pts_y[-indx-2]
                        else:
                            break

                    for indx, x in enumerate(pts_x):
                        if img[0:img.shape[0],x].mean() <= 1:
                            min_x = pts_x[indx+1]
                            assert indx < len(pts_x)
                        else:
                            break
                    for indx, x in enumerate(pts_x):
                        if img[0:img.shape[0], pts_x[-indx-1]].mean() <= 1:
                            if indx == len(pts_x):
                                break
                            max_x = pts_x[-indx-2]
                        else:
                            break
                    pt1_x, pt2_x = numpy.clip([pt1_x, pt2_x], min_x, max_x)
                    pt1_y, pt2_y = numpy.clip([pt1_y, pt2_y], min_y, max_y)

                    #pt1_x, pt2_x = numpy.clip([pt1_x, pt2_x], 0, img.shape[1]-1)
                    #pt1_y, pt2_y = numpy.clip([pt1_y, pt2_y], 0, img.shape[0]-1)

                    box = pt1_x, pt1_y, pt2_x, pt2_y
                    crop_img = img[pt1_y:pt2_y, pt1_x:pt2_x]
                    resized_img = cv2.resize(crop_img, size, interpolation=cv2.INTER_AREA)

                    if what_to_save == 'bbox_coords':
                        raise NotImplementedError('Option %s not implmemented.'%what_to_save)
                        facetubes[clip][clip_i][frame] = list(box)
                    elif what_to_save == 'img':
                        save_name = "%s%s-%d-%03d_.png" % (des_path, clip, clip_i, frame)
                        cv2.imwrite(save_name, resized_img)
                    elif what_to_save == 'mat':
                        save_name = "%s%s-%d-%03d_.mat" % (des_path, clip, clip_i, frame)
                        scipy.io.savemat(save_name, mdict={'img': resized_img})
                    else:
                        raise NotImplementedError('Option %s not implmemented.'%what_to_save)
    return facetubes

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

def save_crop(img, save_path, box, size):
    pt1_x, pt1_y, pt2_x, pt2_y = [int(item) for item in box]
    image = cv2.imread(img)
    #img = Image.open(img)
    #img = img.crop(box)
    #img.save(save_path)
    pt1_x, pt2_x = numpy.clip([pt1_x, pt2_x], 0, image.shape[1]-1)
    pt1_y, pt2_y = numpy.clip([pt1_y, pt2_y], 0, image.shape[0]-1)
    crop_img = image[pt1_y:pt2_y, pt1_x:pt2_x]
    resized_img = cv2.resize(crop_img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path, resized_img)


def draw(img, save_path, boxes, colors):
    img = Image.open(img)
    draw = ImageDraw.Draw(img)
    for box, color in zip(boxes, colors):
        draw.line((box[0], box[1], box[0], box[3]), fill = color, width=5)
        draw.line((box[0], box[1], box[2], box[1]), fill = color, width=5)
        draw.line((box[2], box[1], box[2], box[3]), fill = color, width=5)
        draw.line((box[2], box[3], box[0], box[3]), fill = color, width=5)
    img.save(save_path)


def smoothing(box):
    pt1_x, pt1_y, pt2_x, pt2_y = box

    # Get the largest square coordinates that we can draw in the bounding box.
    # We will take the smallest bounding box's side as the length of the square.
    bb_height = pt2_y - pt1_y
    bb_width = pt2_x - pt1_x
    min_length = min(bb_height, bb_width)
    max_length = max(bb_height, bb_width)
    rem = 1*( (max_length - min_length) / 2. )
    if max_length == bb_height:
        sq_pt1_x = pt1_x
        sq_pt1_y = pt1_y + rem
        sq_pt2_x = pt2_x
        sq_pt2_y = pt2_y - rem
    else:
        sq_pt1_x = pt1_x + rem
        sq_pt1_y = pt1_y
        sq_pt2_x = pt2_x - rem
        sq_pt2_y = pt2_y

    sq_bb_coords = sq_pt1_x, sq_pt1_y, sq_pt2_x, sq_pt2_y
    sq_bb_center_coords = (sq_pt1_x + ((sq_pt2_x - sq_pt1_x) / 2.),  sq_pt1_y + ((sq_pt2_y - sq_pt1_y) / 2.) )
    sq_bb_length = sq_pt2_x - sq_pt1_x
    sq_bb_length = int(sq_pt2_x - sq_pt1_x)

    sq_bb_coords = [int(item) for item in sq_bb_coords]
    sq_bb_center_coords = [int(item) for item in sq_bb_center_coords]

    return sq_bb_coords, sq_bb_center_coords, sq_bb_length
