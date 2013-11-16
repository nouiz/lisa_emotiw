
class emoDict:
    def __init__(self):
        self.emo2numDict = {}
        self.num2emoDict = {}
    
    def readFile(self, file_name):
        file = open(file_name, 'r')
        line = file.readline()
        
        while line != '':
            if line[0] != '=':
                ln = line.split(':')
                emo = ln[0].strip()
                num = ln[1].rstrip('done').strip()
                self.emo2numDict[emo] = num
                self.num2emoDict[num] = emo
            else:
                print 'now in: ', line
    
    def emo2num(self, emo):
        return self.emo2numDict[emo]
    
    def num2emo(self, num):
        return self.emo2numDict[num]
    

if __name__ = '__main__':
    
    file_translate_path = '/data/lisatmp2/emo_video/new_clips'
    extracted_frames_path = '/data/lisatmp2/emo_video/new_clips/ExtractedFrames'
    
    
    emoDic = emoDict()
    emoDic.readFile(file_translate_path)
    
    '''
    extracted_folder_list = glob.glob
    for folder in extracted_folder_list:
        label = check_label(folder)
        
        if label == 'Angry':
        elif label == 'Happy':
        
    
    
    
    
    
    