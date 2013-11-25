function useSVM4(path_in, path_out, clip_id)

%Test
featureTestConvNet = ...
    importfile(fullfile(path_in, [clip_id '_segments.csv']));
input_features = featureTestConvNet(:,1);
dummy_labels = featureTestConvNet(:,2);
featureTestConvNet(:,1:2)=[];

load('svm_results_on_convnet');
[predict_label_test, accuracy_test, prob_values_test] = svmpredict(dummy_labels, input_features, model, '-b 1');

featureTest = featureTestConvNet;

save(fullfile(path_out, ['svm_convnet_pred_' clip_id]),...
    'featureTest' ,'input_features' ,'dummy_labels' ,'prob_values_test' ,'predict_label_test');
end

