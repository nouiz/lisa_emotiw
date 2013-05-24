from face_bbox import *

def test1():
    which_set = "train"
    path = "/data/lisatmp/data/faces_bbox/face_data.h5"
    start = 0
    stop = 1000
    size_of_receptive_field = [64, 64]
    stride = 1
    use_output_map = True
    dataset = FaceBBox(which_set=which_set,
                        start=start,
                        stop=stop,
                        use_output_map=use_output_map,
                        size_of_receptive_field=size_of_receptive_field,
                        path=path)

    import ipdb; ipdb.set_trace()

if __name__=="__main__":
    test1()
