import PIL
import PIL.Image
import numpy.random
import math

def crop_face(img, bbox, eyes, points={}):
    img = img.convert("RGBX")
    bb = bbox
    x0, y0, x1, y1 = (None, None, None, None)
    to_remove = []
    
    for pt in points:
        if math.isnan(points[pt][0]) or math.isnan(points[pt][1]):
            to_remove.append(pt)

    for tr in to_remove:
        del points[tr]
    
    if bb is None:
        try:
            x0, y0 = points['left_eyebrow_outer_end']
            x1, _ = points['right_eyebrow_outer_end']
        except KeyError:
            try:
                if x0 is None:
                    x0, y0 = points['left_eyebrow_center']
                x1, _ = points['right_eyebrow_center']
            except KeyError:
                try:
                    if x0 is None:
                        x0, y0 = points['left_eyebrow_inner_end']
                    x1, _ = points['right_eyebrow_inner_end']
                except KeyError:
                    try:
                        if y0 is None:
                            _, y0 = points['left_eye_center']
                        x1, _ = points['right_mouth_corner']
                        
                        if x0 is None:
                            x0, _ = points['left_mouth_corner']
                    except KeyError:
                        if x0 is None:
                            x0 = 0
                        if y0 is None:
                            y0 = 0                
                        if x1 is None:
                            x1 = img.size[0]-1

        try:
            _, y1 = points['chin_center']
        except KeyError:
            try:
                _, y1 = points['mouth_center']
            except KeyError:
                y1 = img.size[1]-1

    else:
        x0, y0, x1, y1 = tuple(bb)

    x_delta = None #look to the sides more than the keypoints would suggest
    y_delta = None    

    x0 = float(x0)
    x1 = float(x1)
    y0 = float(y0)
    y1 = float(y1)

    try:
        x_delta = eyes[2] - eyes[0]
        y_delta = eyes[3] - eyes[1]

    except TypeError:
        try:
            y_delta = eyes[3] - eyes[1]
        except TypeError:
            if x_delta is None:
                x_delta = img.size[0]/5.0
            if y_delta is None:
                y_delta = img.size[1]/5.0
        if x_delta is None:
            x_delta = img.size[0]/5.0

    x_diff = abs(x1 - x0 + x_delta)
    y_diff = abs(y1 - y0 + y_delta)
    side = min(x_diff + x_delta, y_diff + y_delta)
    
    x0 = max(min(x0, x1) - side/2.0, 0.0)
    x1 = min(x0 + 2.0*side, img.size[0])
    y0 = max(min(y0, y1) - side/2.0, 0.0)
    y1 = min(y0 +  2.0*side, img.size[1])

    side_0 = x1 - x0
    side_1 = y1 - y0

    clamped = [(p, ((96.0*(points[p][0] - x0))/(side_0), (96.0*(points[p][1] - y0))/(side_1))) for p in points if x1 > points[p][0] > x0 and y1 > points[p][1] > y0]

    new_points = {}
    for c in clamped:
        new_points[c[0]] = c[1]
    
    method = None
    if img.size[0] < 96 or img.size[1] < 96:
        method = PIL.Image.ANTIALIAS
    else:
        method = PIL.Image.BICUBIC

    img = img.crop((int(x0), int(y0), int(x1), int(y1)))
    img.load()
    img = img.resize((96, 96), method)

    return (img, new_points)

def display_1(ds, idx):
    img, pts = crop_face(PIL.Image.open(ds.get_original_image_path(idx)), ds.get_bbox(idx), ds.get_eyes_location(idx), ds.get_keypoints_location(idx))
    img = img.convert("RGBA")

    data_str = ""
        
    for y in xrange(96):
        for x in xrange(96):
            if (float(x), float(y)) in pts:
                data_str += '\x00\x00\x00\x00'
            else:
                data_str += '\x00\x00\x00\xff'

    pt_img = PIL.Image.fromstring(mode="RGBA", size=(96, 96), data=data_str)
    img = PIL.Image.composite(img, pt_img, pt_img)
    
    img.show()

def display_n(ds, n):
    for i in xrange(min(n, len(ds))):
        display_1(ds, i)

def display_n_rnd(ds, n):
    rng = numpy.random.RandomState()

    for i in xrange(min(n, len(ds))):
        idx = rng.randint(0, len(ds))
        display_1(ds, idx)
