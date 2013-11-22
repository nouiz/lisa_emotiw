function useSVM4(path_out)
%d = 'C:\challenges\AFEWTest\FINAL3\experiment_4\';

%Test
featureTest = ...
importfile('../working-cuda-convnet-2013-11-15/data/tmp/ConvNet__2013-07-17_13.27.15_segments.csv');
videoIDTest = featureTest(:,1);
labelTest = featureTest(:,2);
featureTest(:,1:2)=[];

load('svm_results_exp4_caglar');

%model = svmtrain(labelTrain,featureTrain ,'-c 18.1 -b 1');
%[predict_label, accuracy, prob_values] = svmpredict(labelVal, featureVal, model, '-b 1');
%accuracy

% predict on test
%[predict_label_train, accuracy_train, prob_values_train] = svmpredict(labelTrain,featureTrain, model, '-b 1');
%[predict_label_val, accuracy_val, prob_values_val] = svmpredict(labelVal,featureVal, model, '-b 1');
%labelTest
%featureTest
%model
[predict_label_test, accuracy_test, prob_values_test] = svmpredict(labelTest,featureTest, model, '-b 1');

save([path_out 'svm_results_exp4'],...
'featureTest' ,'videoIDTest' ,'labelTest' ,'prob_values_test' ,'predict_label_test');
end

