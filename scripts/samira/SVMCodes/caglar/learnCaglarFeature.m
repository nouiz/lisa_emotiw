function learnCaglarFeature()
ccc
load('svm_results_exp4_caglar.mat');

[videoIDTrainAudio,labelschar,preds,f1,f2,f3,f4,f5,f6,f7]  = ...
    textread('caglar_train.txt','%s %s %s %f %f %f %f %f %f %f');
labelTrainAudio = returnLabels(labelschar)';
featuretrainaudio = [f1 f2 f3 f4 f5 f6 f7];
featureTrainAll = concate(videoIDTrain,prob_values_train,...
    videoIDTrainAudio,featuretrainaudio);

[videoIDValidAudio,labelschar,preds,f1,f2,f3,f4,f5,f6,f7]  = ...
    textread('caglar_valid.txt','%s %s %s %f %f %f %f %f %f %f');
labelValAudio = returnLabels(labelschar)';
featurevalidaudio = [f1 f2 f3 f4 f5 f6 f7];
featureValAll = concate(videoIDVal,prob_values_val,...
    videoIDValidAudio,featurevalidaudio);

[videoIDTestAudio,preds,f1,f2,f3,f4,f5,f6,f7]  = ...
    textread('caglar_test.txt','%s %s %f %f %f %f %f %f %f');
labelTestAudio = labelTest;
featuretestaudio = [f1 f2 f3 f4 f5 f6 f7];
featureTestAll = concate(videoIDTest,prob_values_test,...
    videoIDTestAudio,featuretestaudio);

%svm
c_ = 2.^[0:0.2:2];
g_ = 2.^[0:0.2:2];
indx = 0;
for c=c_
    for g=g_
        params = ['-t 2 -g ' num2str(g) ' -c ' num2str(c) ' -b 1'];
        indx = indx+1;
        model = ...
            svmtrain(labelTrainAudio, featureTrainAll,params);
        svmpredict(labelTrainAudio, featureTrainAll, model, '-b 1');
        [predict, accuracyvalid(:,indx), prob]=...
            svmpredict(labelValAudio, featureValAll, model, '-b 1');
        accuracyvalidgc(1,indx)=g; accuracyvalidgc(2,indx)=c;
    end
end

[o l]=max(accuracyvalid(1,:))
g=accuracyvalidgc(1,l); c=accuracyvalidgc(2,l);
params = ['-t 2 -g ' num2str(g) ' -c ' num2str(c) ' -b 1'];
model = ...
    svmtrain(labelTrainAudio, featureTrainAll,params);
svmpredict(labelTrainAudio, featureTrainAll, model, '-b 1');
[predict, accuracy, prob]=...
    svmpredict(labelValAudio, featureValAll, model, '-b 1');

% predict on all
[predict_label_trainf, accuracy_trainf, prob_values_trainf] = ...
svmpredict(labelTrainAudio,featureTrainAll, model, '-b 1');

[predict_label_valf, accuracy_valf, prob_values_valf] = ...
svmpredict(labelValAudio,featureValAll, model, '-b 1');

[predict_label_testf, accuracy_testf, prob_values_testf] = ...
svmpredict(labelTestAudio,featureTestAll, model, '-b 1');
save(['svm_results_exp4_audio_concat'],...
'featureTrainAll','videoIDTrainAudio','labelTrainAudio','prob_values_trainf','predict_label_trainf','accuracy_trainf',...
'featureValAll'  ,'videoIDValidAudio'  ,'labelValAudio'  ,'prob_values_valf'  ,'predict_label_valf'  ,'accuracy_valf',...
'featureTestAll' ,'videoIDTestAudio' ,'labelTestAudio' ,'prob_values_testf' ,'predict_label_testf' ,'model'...
    );

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