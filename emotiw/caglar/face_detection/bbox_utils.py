import numpy
import math


def get_area_ratio(width1, height1, width2, height2):
    s1 = width1*height1
    s2 = width2*height2
    return s1/s2

def get_rf_end_loc(img_shape, conv_out_shp, stride, r, c):
    rf_x_no_ = int(math.ceil(c/stride))
    rf_y_no_ = int(math.ceil(r/stride))
    rf_x_no = rf_x_no_ if rf_x_no_ < conv_out_shp[1] else conv_out_shp[1]-1
    rf_y_no = rf_y_no_ if rf_y_no_ < conv_out_shp[0] else conv_out_shp[0]-1
    return rf_x_no, rf_y_no

def get_rf_start_loc(img_shape, rf_shape, stride, r, c):
    rf_x_no_ = int(math.floor(c/stride) - math.ceil(rf_shape[1]/stride))
    rf_y_no_ = int(math.floor(r/stride)- math.ceil(rf_shape[1]/stride))
    rf_x_no = rf_x_no_ if rf_x_no_ > 0 else 0
    rf_y_no = rf_y_no_ if rf_y_no_ > 0 else 0
    return rf_x_no, rf_y_no

def get_conv_out_size(img_shape, rf_shape, stride):
    conv_out_x_size = int((img_shape[1] - rf_shape[1])/stride) + 1
    conv_out_y_size = int((img_shape[0] - rf_shape[0])/stride) + 1
    return conv_out_x_size, conv_out_y_size

def perform_hit_test(bbx_start, h, w, point):
    """
        Check if a point is in the bounding box.
    """
    if (bbx_start[0] <= point[0] and bbx_start[0] + h >= point[0]
            and bbx_start[1] + w >= point[1] and bbx_start[1] <= point[1]):
        return True
    else:
        return False

def get_image_bboxes(image_index, bboxes):
    """
        Query pytables table for the given range of images.
    """

    start = image_index.start
    stop = image_index.stop
    query = "(imgno>={}) & (imgno<{})".format(start, stop)
    bboxes_result = bboxes.readWhere(query)
    #if bboxes_result.shape[0] == 131:
    #    import ipdb; ipdb.set_trace()
    return bboxes_result

def convert_bboxes_exhaustive(bbx_targets, img_shape, rf_shape, area_ratio, stride=1):
    """
    This function converts the bounding boxes to the spatial outputs for the neural network.
    In order to do this, we do a naive convolution and check if the bounding box is inside a
    given receptive field.
    Parameters
    ---------

    bbx_targets: pytables table.
    img_shape: list
    rf_shape: list
    stride: integer
    """
    assert bbx_targets is not None
    assert img_shape is not None
    assert rf_shape is not None

    prev_rec = None
    output_maps = []
    batch_size = bbx_targets.shape[1]

    rs = bbx_targets[:]["row"][0]
    cs = bbx_targets[:]["col"][0]
    widths = bbx_targets[:]["width"][0]
    heights = bbx_targets[:]["height"][0]
    imgnos = bbx_targets[:]["imgno"][0]

    conv_out_x_size, conv_out_y_size = get_conv_out_size(img_shape, rf_shape, stride)

    for i in xrange(batch_size):
        rf_y_start = 0
        rf_y_end = rf_shape[0]

        issame = False
        r = rs[i]
        c = cs[i]
        width = widths[i]
        height = heights[i]
        imgno = imgnos[i]

        if prev_rec is not None:
            if prev_rec == imgno:
                issame = True
                output_map = output_maps[-1]
            else:
                issame = False
                output_map = numpy.zeros((conv_out_y_size*conv_out_x_size))
        else:
            output_map = numpy.zeros((conv_out_y_size*conv_out_x_size))

        area_ratio_ = get_area_ratio(rf_shape[1], rf_shape[0], width, height)

        if area_ratio < area_ratio_:
            if not issame:
                prev_rec = imgno
                #output_map = [0] * conv_out_y_size * conv_out_x_size
                output_maps.append(output_map)
            continue

        out_idx = 0
        #Perform convolution for each bounding box
        while (rf_y_end <= (img_shape[0] - rf_shape[0])):
            rf_x_start = 0
            rf_x_end = rf_shape[1]
            while (rf_x_end <= (img_shape[1] - rf_shape[1])):
                s_w = 0
                s_h = 0

                #Check if any corner of the image falls inside the boundary box:
                if perform_hit_test([rf_y_start, rf_x_start], rf_shape[0], rf_shape[1], [r, c]):
                    x2 = min(rf_x_end, c + width)
                    y2 = min(rf_y_end, r + height)
                    s_w = x2 - c
                    s_h = y2 - r
                elif perform_hit_test([rf_y_start, rf_x_start], rf_shape[0], rf_shape[1], [r + height, c]):
                    x2 = min(rf_x_end, c + width)
                    y2 = r + height
                    s_w = x2 - c
                    s_h = y2 - rf_y_start
                elif perform_hit_test([rf_y_start, rf_x_start], rf_shape[0], rf_shape[1], [r,c+width]):
                    x2 = c + width
                    y2 = min(rf_y_end, r + height)
                    s_w = x2 - rf_x_start
                    s_h = y2 - r
                elif perform_hit_test([rf_y_start, rf_x_start], rf_shape[0], rf_shape[1], [r+height,c+width]):
                    x2 = c + width
                    y2 = r + height
                    s_w = x2 - rf_x_start
                    s_h = y2 - rf_y_start

                #import ipdb; ipdb.set_trace()
                s_area = s_w * s_h
                area = width * height

                #print area, s_area
                #If the face area is very small ignore it.
                if area <= 18 or s_area <= 18:
                    ratio = 0.
                else:
                    ratio = float(s_area) / float(area)

                #Compare with the previous record
                if not issame:
                    if ratio >= area_ratio:
                        output_map[out_idx] = 1
                    else:
                        output_map[out_idx] = 0
                else:
                    if ratio >= area_ratio:
                        #We don't have +1 here because index starts from 0.
                        #But normally it is (N-M)/stride + 1
                        output_map[out_idx] = 1

                out_idx += 1
                rf_x_start += stride
                rf_x_end = rf_x_start + rf_shape[1]

            rf_y_start += stride
            rf_y_end = rf_y_start + rf_shape[0]

        prev_rec = imgno

        if not issame:
            output_maps.extend([output_map])

    #output_maps = numpy.reshape(output_maps, newshape=(batch_size, -1))
    output_maps = numpy.asarray(output_maps)
    return output_maps

def convert_bboxes_guided(bbx_targets, img_shape, rf_shape, area_ratio, stride=1, conv_outs=False,
        n_channels=None):
    """
    This function converts the bounding boxes to the spatial outputs for the neural network.
    In order to do this, we do a naive convolution and check if the bounding box is inside a
    given receptive field.
    Parameters
    ---------
        bbx_targets: pytables table.
        img_shape: list
        rf_shape: list
        stride: integer
    """
    assert bbx_targets is not None
    assert img_shape is not None
    assert rf_shape is not None
    #print "Stride is, ", stride
    prev_rec = None
    output_maps = []
    batch_size = bbx_targets.shape[1]

    rs = bbx_targets[:]["row"][0]
    cs = bbx_targets[:]["col"][0]
    widths = bbx_targets[:]["width"][0]
    heights = bbx_targets[:]["height"][0]
    imgnos = bbx_targets[:]["imgno"][0]

    conv_out_x_size, conv_out_y_size = get_conv_out_size(img_shape, rf_shape, stride)
    #print "Size %d, %d\n" % (conv_out_x_size, conv_out_y_size)

    for i in xrange(batch_size):
        issame = False
        r = rs[i]
        c = cs[i]
        width = widths[i]
        height = heights[i]
        imgno = imgnos[i]

        rf_x_start_no, rf_y_start_no = get_rf_start_loc(img_shape, rf_shape, stride, r, c)
        rf_x_end_no, rf_y_end_no = get_rf_end_loc(img_shape, [conv_out_y_size, conv_out_x_size], stride, r + height, c + width)

        rf_y_start = stride * rf_y_start_no
        rf_y_end = stride * rf_y_end_no

        rf_x_start = stride * rf_x_start_no
        rf_x_end = rf_x_end_no * stride

        if prev_rec is not None:
            if prev_rec == imgno:
                issame = True
                output_map = output_maps[-1]
            else:
                issame = False
                output_map = numpy.zeros((conv_out_y_size*conv_out_x_size))
        else:
            output_map = numpy.zeros((conv_out_y_size*conv_out_x_size))

        area_ratio_ = get_area_ratio(rf_shape[1], rf_shape[0], width, height)

        if area_ratio > area_ratio_:
            if not issame:
                prev_rec = imgno
                output_maps.append(output_map.flatten())
            continue

        rf_y_iter = rf_y_start
        y_idx = 0

        while(rf_y_iter <= rf_y_end):
            x_idx = 0
            rf_x_iter = rf_x_start
            while(rf_x_iter <= rf_x_end):
                s_w = 0
                s_h = 0
                rf_x_bound = rf_x_iter + rf_shape[1]
                rf_y_bound = rf_y_iter + rf_shape[0]

                #Check if any corner of the image falls inside the boundary box:
                if perform_hit_test([rf_y_iter, rf_x_iter], rf_shape[0], rf_shape[1], [r, c]):
                    x2 = min(rf_x_bound, c + width)
                    y2 = min(rf_y_bound, r + height)
                    s_w = x2 - c
                    s_h = y2 - r

                elif perform_hit_test([rf_y_iter, rf_x_iter], rf_shape[0], rf_shape[1], [r + height, c]):
                    x2 = min(rf_x_bound, c + width)
                    y2 = r + height
                    s_w = x2 - c
                    s_h = y2 - rf_y_iter

                elif perform_hit_test([rf_y_iter, rf_x_iter], rf_shape[0], rf_shape[1], [r,c+width]):
                    x2 = c + width
                    y2 = min(rf_y_end, r + height)
                    s_w = x2 - rf_x_iter
                    s_h = y2 - r

                elif perform_hit_test([rf_y_iter, rf_x_iter], rf_shape[0], rf_shape[1], [r+height,c+width]):
                    x2 = c + width
                    y2 = r + height
                    s_w = x2 - rf_x_iter
                    s_h = y2 - rf_y_iter

                s_area = s_w * s_h
                area = width * height
                #if imgno == 38:
                #    import ipdb; ipdb.set_trace()
                #If the face area is very small ignore it.
                if area <= 18 or s_area <= 18:
                    ratio = 0.
                else:
                    ratio = float(s_area) / float(area)

                out_idx = (rf_y_start_no + y_idx) * conv_out_x_size + rf_x_start_no + x_idx

                #Compare with the previous record
                if not issame:
                    if ratio >= area_ratio_:
                        output_map[out_idx] = 1
                        #print "Area %d, S area %d, imgno %d, loc %d" % (area, s_area, imgno,
                        #        out_idx)
                    else:
                        output_map[out_idx] = 0
                else:
                    if ratio >= area_ratio_:
                        output_map[out_idx] = 1
                        #print "Area %d, S area %d, imgno %d, loc %d" % (area, s_area, imgno,
                        #        out_idx)

                x_idx += 1
                rf_x_iter += stride
            y_idx += 1
            rf_y_iter += stride

        prev_rec = imgno
        if not issame:
            output_maps.append(output_map)

    output_maps = numpy.asarray(output_maps)

    #if output_maps.shape[0] < 100:
    #    import ipdb; ipdb.set_trace()

    if conv_outs:
        output_maps = output_maps.reshape(output_maps.shape[0], 1, conv_out_x_size, conv_out_y_size)
        if n_channels is not None:
            output_maps = numpy.repeat(output_maps, n_channels, axis=1)
    return output_maps


