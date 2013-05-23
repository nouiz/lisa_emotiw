from PIL import Image
import ImageDraw
import ImageFont
import tarfile
from math import isnan

def displayPoint(draw, x, y, idx=0, num_pt=1, radius=2.0):
    ratio = float(idx)/float(num_pt)
    color_val = int(ratio*255)
    fill = (255 - color_val, (color_val * num_pt//2) % 255, color_val)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill)

def displayNum(draw, x, y, idx=0, num_pt=1, size=8): 
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSerif.ttf', int(size))
    except IOError:
        font = ImageFont.load_default()

    displayPoint(draw, x, y, idx, num_pt, radius=2.0)
    draw.text((x-3.0, y+3.0), str(idx), font=font)

def displayImageWithBoundingbox(img, bounds, fn=None):
    """
    Displays the given image with bounds superimposed.
    Expects [x_center, y_center, width, height].
    Fn argument is ignored.
    """   
    
    cp = img.copy().convert("RGB")
    draw = ImageDraw.Draw(cp)

    fill = (0, 255, 0)
    x_center = int(float(bounds[0]))
    y_center = int(float(bounds[1]))
    width = int(float(bounds[2]))
    height = int(float(bounds[3]))
    
    draw.rectangle([(x_center + width/2, y_center + height/2),
                    (x_center - width/2, y_center - height/2)])

    del draw
    cp.show()

def displayImageWithKeypoints(img, keypoints, fn=displayPoint):
    """
    Displays the given image with keypoints superimposed.
    Expects flattened list of x, y points, each pair being
    a different point.
    Fn argument is the function to use to draw points. It
    must accept 6 arguments, and must take the surface,
    x and y coordinates, index of this point, and number
    of points total arguments in that order.
    """
    
    cp = img.copy().convert("RGB")
    draw = ImageDraw.Draw(cp)
    
    for i in xrange(len(keypoints)//2):
        if isnan(float(keypoints[i*2])) or isnan(float(keypoints[i*2+1])):
            continue

        fn(draw, int(float(keypoints[i*2])), int(float(keypoints[i*2+1])), i, len(keypoints)//2)

    del draw
    cp.show()

