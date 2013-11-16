"""
Extracts frame images from avi files
"""

import subprocess
import glob
import os
import sys
#import ipdb

def get_output_size(path, width = 1024):
    """
    Read the aspect ration information and return
    an output size string based on aspect ratio
    and given width

    Params
    -----
    path: avi file path
    widht: output image width
    """

    command = ["ffprobe", path]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = p.communicate()[0]
    asr = res[res.find("DAR "):].split(']')[0][4:].split(':')
    try:
        asr = map(float, asr)
    except ValueError:
        asr = res[res.find("DAR "):].split(' ')[1].split(':')
        asr[1] = asr[1].split(",")[0]
        asr = map(float, asr)

    height = int(width / (asr[0]/asr[1]))
    return "{}x{}".format(width, height)


def extract_frames(src, dest, asr):
    """
    Extract frame images from avi

    Params
    -----
    src: src file
    dest: dest file pattern
    asr: aspect ration string e.g. 1024x576
    """
    print src
    print asr
    print dest
    
    command = ["ffmpeg", "-i", src, "-s", asr, "-qscale", "1", dest]
    #ipdb.set_trace()
    subprocess.call(command)
    
if __name__ == "__main__":

    #_, which = sys.argv

    #if which == 'AFEW2':
    
    
    avi_path = "/data/lisatmp2/EmotiWTest/Test_Vid_Distr/Data"
    img_path = "/data/lisatmp2/EmotiWTest/Test_Vid_Distr/ExtractedFrame"
    report_path = '/data/lisatmp2/EmotiWTest/Test_Vid_Distr'
    
    
    #avi_path = "/Users/zhenzhou/Desktop/UbisoftChallenge/facetube/Data/clips"
    #img_path = "/Users/zhenzhou/Desktop/UbisoftChallenge/facetube/Data/clips"
    #report_path = "/Users/zhenzhou/Desktop/UbisoftChallenge/facetube/Data/report"
    
    
    
    #emots = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "NoClass"]
    #emots = ["Happy"]
    #emots.sort()
        
    extractReport = open(report_path + '/extractReport.txt', 'w')
    
    #for e in range(0, emots.__len__()):
        #file_list = glob.glob("{}/{}/*.avi".format(avi_path, emots[e]))
    file_list = glob.glob("{}/000*.avi".format(avi_path))
    #print "{}/{}*.avi".format(avi_path, emots[e].lower())
    file_list.sort()
    #fileTranslate.write("====< " + emots[e] + " >====" + '\n')
    
    error = 0
    for f in range(0, file_list.__len__()):
        try:
            aviName = file_list[f].split('/')[-1].rstrip('.avi')
            #outputName = "{}{:06d}".format(emo,f+1)
            
            # make path if not exist                
            save_path = "{}/{}".format(img_path, aviName)
        
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
                            
            print file_list[f], '\n'
            print save_path, '\n'
            output = "{}/{}-%3d_.png".format(save_path, aviName)
            # get proper size of image from aspect ration info
            print 'get aspect ratio'
            asr = get_output_size(file_list[f])         
	    print 'asr: ', asr, '\n'
            print 'extract frames'
            extract_frames(file_list[f], output, asr)
            extractReport.write(aviName + ' done' + '\n')
            print aviName + ' done' + '\n'
        except:
            error += 1
            extractReport.write(aviName + ' failed' + '\n')
            print aviName + ' failed' + '\n'
    if error == 0:
        print 'All Extracted Successfully'
        extractReport.write('All Extracted Successfully')
    else:
        print 'Finished Running with errors'
        extractReport.write('Finished Running with errors')
    
