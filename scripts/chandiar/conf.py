config = {
    # What do you want to save:
    # 'img': smoothed facetubes saved as png
    # 'mat': smoothed facetubes saved as mat
    # 'bbox_coords' : smoothed bounding boxes coordinates as pkl !!! NOT IMPLEMENTED!!!
    'what_to_save' : 'img',

    #### INPUTS ####
    'extracted_frames_path' : '/data/lisatmp2/EmotiWTest/Test_Vid_Distr/ExtractedFrame/000143240',
    # Picasa bounding boxes coordinates as .txt files
    'picasa_bbox_path' : '/data/lisatmp2/EmotiWTest/Test_Vid_Distr/BoundBoxData/000143240',
    ################

    #### OUTPUT ####
    # Directory where the smoothed facetubes will be saved.
    'save_path' : '/data/lisa/exp/chandiar/challenge_scripts/sandbox/Challenge/EmotiWTest/smooth_picasa_face_tubes_96_96/images/000143240',
    ################

    # Pixel resolution of the smoothed facetubes.
    # Default is 96 pixels by 96 pixels.
    'size' : (96, 96),

    # Bounding boxes parameters 
    'size_thre' : 10,
    'distance_thr' : .008,
    'overlap_thr' : 0.08,
    'similar_thr' : 0.97,
}
