function generatesunmission()
ccc
load('svm_results_exp4_audio_concat.mat')
for i=1:length(videoIDTestAudio)
    videoIDTestAudio(i)
    
    fid = fopen(['submission2all/' char(videoIDTestAudio(i)) '.txt'], 'w');
    fwrite(fid, returnlabel(predict_label_testf(i)));
    fclose(fid);
end
end

function lb=returnlabel(lbnum)
switch lbnum
    case 0
        lb = 'Angry';
    case 1
        lb = 'Disgust';
    case 2
        lb = 'Fear';
    case 3
        lb = 'Happy';
    case 4
        lb = 'Sad';
    case 5
        lb = 'Surprise';
    case 6
        lb = 'Neutral';
end
end