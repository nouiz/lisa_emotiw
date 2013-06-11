import PIL
import PIL.Image
import numpy.random
import math

def crop_face(img, bbox, eyes, points={}):
    img = img.convert("RGB")
    bb = bbox
    x0, y0, x1, y1 = (None, None, None, None)
    to_remove = []
    
    for pt in points:
        if math.isnan(points[pt][0]) or math.isnan(points[pt][1]):
            to_remove.append(pt)

    for tr in to_remove:
        del points[tr]

    if bb is None:
        right = -1
        left = img.size[0]+1
        top = -1
        bottom = img.size[1]+1
        
        for pt in points:
            right = max(right, points[pt][0])
            left = min(left, points[pt][0])
            bottom = min(bottom, points[pt][1])
            top = max(top, points[pt][1])

        x0 = left
        x1 = right
        y0 = top
        y1 = bottom

    else:
        x0, y0, x1, y1 = tuple(bb)

    x0 = float(x0)
    x1 = float(x1)
    y0 = float(y0)
    y1 = float(y1)

    side_x = abs(x1 - x0)
    side_y = abs(y1 - y0)
    the_side = max(1.3*side_x, 1.3*side_y)
    
    side_x = min(img.size[0], the_side)
    side_y = min(img.size[1], the_side)

    x0 = max(min(x0, x1) - side_x/2.0, 0)
    x1 = min(x0 + 1.5*side_x, img.size[0])
    y0 = max(min(y0, y1) - side_y/2.0, 0)
    y1 = min(y0 + 1.5*side_y, img.size[1])

    side_x = x1 - x0
    side_y = y1 - y0
    the_side = min(side_x, side_y)

    clamped = [(p, ((96.0*(points[p][0] - x0))/(the_side), (96.0*(points[p][1] - y0))/(the_side))) for p in points if x1 >= points[p][0] >= x0 and y1 >= points[p][1] >= y0]

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
    img0 = PIL.Image.open(ds.get_original_image_path(idx))

    data_str = ""
        
    for y in xrange(96):
        for x in xrange(96):
            found = False
            for p in pts:
                if int(pts[p][0]) == x and int(pts[p][1]) == y:
                    data_str += '\x00\x00\x00\x00'
                    found = True
                    break
            if not found:
                data_str += '\x00\x00\x00\xff'

    pt_img = PIL.Image.fromstring(mode="RGBA", size=(96, 96), data=data_str)
    img = PIL.Image.composite(img, pt_img, pt_img)
    img0.paste(img, (img0.size[0]-96, img0.size[1]-96, img0.size[0], img0.size[1]))
    
    img0.show()

def display_n(ds, n):
    for i in xrange(min(n, len(ds))):
        display_1(ds, i)

def display_n_rnd(ds, n):
    rng = numpy.random.RandomState()

    for i in xrange(min(n, len(ds))):
        idx = rng.randint(0, len(ds))
        display_1(ds, idx)
