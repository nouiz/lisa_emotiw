function useSVM4(path_in, path_out, clip_id)

load('svm_results_on_convnet');

% override featureTest and labelTest that were loaded from svm_results_on_convnet.mat
featureTest = ...
    importfile(fullfile(path_in, [clip_id '_segments.csv']));
videoIDTest = featureTest(:,1);
labelTest = featureTest(:,2);
featureTest(:,1:2)=[];

[predict_label_test, accuracy_test, prob_values_test] = svmpredict(labelTest, featureTest, model, '-b 1');

save(fullfile(path_out, ['svm_convnet_pred_' clip_id]),...
    'featureTest' ,'videoIDTest' ,'labelTest' ,'prob_values_test' ,'predict_label_test');
end

