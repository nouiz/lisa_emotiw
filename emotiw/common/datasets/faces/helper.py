import os

def flatten(points):
    return [item for pair in points for item in pair] 

def within_bounds(idx, max_bound, min_bound=None):
    if min_bound is None:
        min_bound = idx

    if idx >= max_bound or idx < min_bound:
        return False
    return True

def read_points(lst_images, transform_rule, directory=None):
    all_points = []

    if directory is None:
        directory = '.'

    for img in lst_images:
        points = []
        f = open(os.path.join(directory, transform_rule(img)))
        for l in f.readlines()[3:-1]:
            x_y = l.split(' ')
            points.extend([float(x_y[0]), float(x_y[1])])
        all_points.append(points)
        f.close()

    return all_points

