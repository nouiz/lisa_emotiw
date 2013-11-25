function learnCaglarFeature(prediction_path, clip_id)

load('svm_results_on_convnet_and_audio.mat');
trainedmodel=model;

load(fullfile(prediction_path, ['svm_convnet_pred_' clip_id '.mat']));

[videoIDTestAudio,preds,f1,f2,f3,f4,f5,f6,f7]  = ...
    textread(fullfile(prediction_path, ['audio_pred_' clip_id '.txt']),'%s %s %f %f %f %f %f %f %f')
%labelTestAudio = labelTest
labelTestAudio = [0]
featuretestaudio = [f1 f2 f3 f4 f5 f6 f7]
featureTestAll = concate(videoIDTest,prob_values_test,...
    videoIDTestAudio,featuretestaudio)

[predict_label_test, accuracy_test, prob_values_test] = ...
    svmpredict(labelTestAudio,featureTestAll, model, '-b 1')

save(fullfile(prediction_path, ['svm_convnet_audio_pred_' clip_id]),...
    'featureTestAll' ,'videoIDTestAudio' ,'labelTestAudio' ,'prob_values_test' ,'predict_label_test')

end

function lb = returnLabels(labels)
for i = 1:size(labels,1)
    switch char(labels(i))
        case 'Angry'
            lb(i) = 0;
        case 'Disgust'
            lb(i) = 1;
        case 'Fear'
            lb(i) = 2;
        case 'Happy'
            lb(i) = 3;
        case 'Sad'
            lb(i) = 4;
        case 'Surprise'
            lb(i) = 5;
        case 'Neutral'
            lb(i) = 6;
    end
end
end

function finalfeatures = concate(id1,fe1,id2,fe2)
finalfeatures = [];
for j=1:length(id2)
    flag=1;
    for i=1:length(id1)
        ind = [];
        ind = findstr(num2str(id1(i)), char(id2(j)));
        if ~isempty(ind)
            finalfeatures(j,:)=[fe1(i,:) fe2(j,:)];
            flag=0;
            break;
        end
    end
    if flag==1
        finalfeatures(j,:)=[0 0 0 0 0 0 0 fe2(j,:)];
    end
end
end
