
#from Challenge.trainingsetScripts.emoDict import emoDict


if __name__ == '__main__':
    path = '/Users/zhenzhou/Desktop/'
    
    file = open(path+'overlap.txt', 'r')
    line = file.readline()
    dict = {}
    
    min = 1
    max = 240
    face = 'face4'
    
    report = open(path + face +'.txt', 'w')
    report.write(face + ':' + str(min) + ':' + str(max-1) + '\n')
    while line != '':
        str = line.split('\t')
        #print line
        #print str
        arr1 = str[0].split(':')[0].strip()
        arr2 = str[1].split(':')[0].strip()

        #print 'len:', len(str)
        #print 'arr1:', arr1
         
        if len(arr2) > 0:
            #   print arr1
            #print 'arr2:', arr2
          
            dict[arr1] = arr2
            dict[arr2] = arr1
        elif len(arr2) == 0:
            dict[arr1] = 0
        line = file.readline()
            
    #print dict
    for i in range(min, max):
        noclass =  "noclass{}".format(i)
        #print dict['noclass4']
        if not dict.get(noclass) is None:    
            val = dict[noclass]
            #print 'value', dict[noclass]
            if val == 0:
                #report.write('no_conflict:' + noclass)
                print 'no_conflict:' + noclass
            else:
                num = int(val[7:])
                if num in range(min, max):
                    report.write(val + ':' + noclass +'\n')
                    print 'conflict:' + val + ':' + noclass
                    
                else:
                    #report.write('no_conflict:' + noclass)
                    print 'no_conflict:' + noclass
        else:
            print noclass, 'does not exist'
    

#def rm_conflict_frames(batch_path, conflict_report):
    
            
            
        