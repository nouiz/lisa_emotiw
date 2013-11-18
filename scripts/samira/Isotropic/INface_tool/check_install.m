% This scripts checks the installation of the INface toolbox v2.0. 
% 
% All this script does is running a test using all (or at least most) of the 
% functions featured in the INface toolbox v2.0. You can run this script to 
% check, whether the installation script has successfully added all directories
% and subderectories of the toolboy to Matlabs sesrch paths or if you have
% installed the toolbox manually, whether your manual installation was
% successful.
% 
% The script can also be used to check whether the c/c++ code was compiled
% successfully.
% 
% It should be noted that all components of the toolbox were tested with 
% Matlab version 7.5.0.342 (R2007b) and WindowsXP Professional (SP3) OS.
% 
% NOTE: This script assumes that a 128x128 pixel grey-scale image 
% called "sample_image.bmp" is located in Matlabs path. The image is
% usually distibuted with the toolbox, so make sure you do not accidentally
% delete it.
% 
% COMMENT: Note then this script can take a while, since its original
% purpose was to error-check the scripting of the toolbox. Hence, it tests
% most of the functions (i.e., all functions that are also usefull on their 
% own) of the toolbox with a variety of input parameter combinations. 

%% Some init operations
clear all
close all

report.msg      = cell(1,20);
report.ok       = zeros(1,20);
report.test_no  = 0;

%% Read test image
disp('Reading test image to check installation...')
X=imread('sample_image.bmp');
disp('Done.')
disp(' ')

%% Testing functions from "auxilary" folder

disp('Testing independant functions from "auxilary" folder.')
ok = 1;

%adjust range
report.test_no = report.test_no + 1;
try
  Y=adjust_range(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=adjust_range(X,[0,1]);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=adjust_range(X,[0,0.1]);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=adjust_range(X,[0 255]);
  if(size(Y)~=size(X))
      ok=0;
  end
  if(ok)
    report.msg{report.test_no} = 'Function "adjust_range" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "adjust_range" is NOT working properly.';
    report.ok(report.test_no)  = 0; 
  end
      
catch
  report.msg{report.test_no} = 'Function "adjust_range" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end



%gamma_correction
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=gamma_correction(X, [0 1], [0 1], 0.2);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=gamma_correction(X, [1 255], [0 255], 0.4);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=gamma_correction(X, [], [0 255], 0.4);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "gamma_correction" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "gamma_correction" is NOT working properly.';
    report.ok(report.test_no)  = 0;   
  end
catch
  report.msg{report.test_no} = 'Function "gamma_correction" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end




%threshold_filtering
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=threshold_filtering(X, 9, 1);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=threshold_filtering(X, [], 5);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=threshold_filtering(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "threshold_filtering" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "threshold_filtering" is NOT working properly.';
    report.ok(report.test_no)  = 0;   
  end
catch
  report.msg{report.test_no} = 'Function "threshold_filtering" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end





%normalize8
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=normalize8(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=normalize8(X,1);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=normalize8(X,0);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
      report.msg{report.test_no} = 'Function "normalize8" is working properly.';
      report.ok(report.test_no)  = 1;
  else
      report.msg{report.test_no} = 'Function "normalize8" is NOT working properly.';
      report.ok(report.test_no)  = 0; 
  end
catch
  report.msg{report.test_no} = 'Function "normalize8" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end

disp('Done.')

%% Testing functions from "histograms" folder

disp('Testing independant functions from "histograms" folder.')

%fitt_distribution
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=fitt_distribution(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=fitt_distribution(X,2);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=fitt_distribution(X,2,[0 0.2]);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=fitt_distribution(X,3,0.005);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "fitt_distribution" is working properly.';
    report.ok(report.test_no)  = 1;
  else
     report.msg{report.test_no} = 'Function "fitt_distribution" is NOT working properly.';
     report.ok(report.test_no)  = 0;   
  end
catch
  report.msg{report.test_no} = 'Function "fitt_distribution" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end


%rank_normalization
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=rank_normalization(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=rank_normalization(X,'two');
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=rank_normalization(X,'three','descend');
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "rank_normalization" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "rank_normalization" is NOT working properly.';
    report.ok(report.test_no)  = 0;  
  end
catch
  report.msg{report.test_no} = 'Function "rank_normalization" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end

disp('Done.')

%% Testing functions from "postprocessors" folder

disp('Testing independant functions from "postprocessors" folder.')


%histtruncate
ok = 1;
report.test_no = report.test_no + 1;
try
  [Y, dummy] =histtruncate(Y, 0.2, 0.2);
  if(size(Y)~=size(X))
      ok=0;
  end
  if(ok)
    report.msg{report.test_no} = 'Function "histtruncate" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "histtruncate" is NOT working properly.';
    report.ok(report.test_no)  = 0;  
  end
catch
  report.msg{report.test_no} = 'Function "histtruncate" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end


%robust_postprocessor
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=robust_postprocessor(X, 0.1, 10);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=robust_postprocessor(X, 0.3);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=robust_postprocessor(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "robust_postprocessor" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "robust_postprocessor" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "robust_postprocessor" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end

disp('Done.')


%% Testing functions from "photometric" folder

disp('Testing independant functions from "photometric" folder. This may take while.')


%dog filter
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = dog(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = dog(X,1, 2);
  if(size(Y)~=size(X))
      ok=0;
  end
   Y = dog(X,1, 2,0);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  Y = dog(X,[],[],1);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "dog" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "dog" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "dog" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end


%single_scale_retinex
ok = 1;
report.test_no = report.test_no + 1;
try
  R=single_scale_retinex(X);
  if(size(R)~=size(X))
      ok=0;
  end
 R=single_scale_retinex(X,9);
 if(size(R)~=size(X))
      ok=0;
  end
  R=single_scale_retinex(X,9,0);
  if(size(R)~=size(X))
      ok=0;
  end
  
  R=single_scale_retinex(X,[],0);
  if(size(R)~=size(X))
      ok=0;
  end
  
  [R,L] = single_scale_retinex(X,[],1);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "single_scale_retinex" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "single_scale_retinex" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "single_scale_retinex" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end

%tantriggs
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = tantriggs(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = tantriggs(X,0.7,0);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = tantriggs(X,0.3);
  if(size(Y)~=size(X))
      ok=0;
  end 
  Y = tantriggs(X,[],0);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "tantriggs" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "tantriggs" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "tantriggs" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end


%weberfaces
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = weberfaces(X,1, 9, 2, 0);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = weberfaces(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = weberfaces(X,[], 25, 2);
  if(size(Y)~=size(X))
      ok=0;
  end 
  Y = weberfaces(X,[], [], [], 0);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "weberfaces" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "weberfaces" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "weberfaces" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end




%multi scale weberfaces
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = multi_scale_weberfaces(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = multi_scale_weberfaces(X,[1 0.5], [9 49], [4 0.04], [0.5 1], 1);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = multi_scale_weberfaces(X,[1 0.25], [9 25], [2 0.4], [1 0.5], 0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "multi_scale_weberfaces" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "multi_scale_weberfaces" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "multi_scale_weberfaces" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end






%gradientfaces
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = gradientfaces(X,0.8, 1);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = gradientfaces(X,[], 0);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = gradientfaces(X);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  Y = gradientfaces(X,1);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "gradientfaces" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "gradientfaces" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "gradientfaces" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end




%anisotropic_smoothing_stable
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = anisotropic_smoothing_stable(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = anisotropic_smoothing_stable(X,20);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  [Y,L] = anisotropic_smoothing_stable(X);
  if(size(Y)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  
  [Y,L] = anisotropic_smoothing_stable(X,[],0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "anisotropic_smoothing_stable" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "anisotropic_smoothing_stable" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "anisotropic_smoothing_stable" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end



%DCT_normalization
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=DCT_normalization(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=DCT_normalization(X,40);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  Y=DCT_normalization(X,[], 0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  Y=DCT_normalization(X,100, 0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "DCT_normalization" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "DCT_normalization" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "DCT_normalization" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end



%lssf_norm
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = lssf_norm(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = lssf_norm(X, 0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "lssf_norm" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "lssf_norm" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "lssf_norm" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end





%adaptive_nl_means_normalization
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = adaptive_nl_means_normalization(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = adaptive_nl_means_normalization(X,20);
  if(size(Y)~=size(X))
      ok=0;
  end 
  Y = adaptive_nl_means_normalization(X,150,3);
  if(size(Y)~=size(X))
      ok=0;
  end
  [R,L] = adaptive_nl_means_normalization(X,150,[],0);
  if(size(R)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  [R,L] = adaptive_nl_means_normalization(X,[],[],0);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end 
  [R,L] = adaptive_nl_means_normalization(X,150,3,1);
  if(size(R)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "adaptive_nl_means_normalization" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "adaptive_nl_means_normalization" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "adaptive_nl_means_normalization" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end



%adaptive_single_scale_retinex
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = adaptive_single_scale_retinex(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = adaptive_single_scale_retinex(X,15);
  if(size(Y)~=size(X))
      ok=0;
  end 
  Y = adaptive_single_scale_retinex(X,10,10,1);
  if(size(Y)~=size(X))
      ok=0;
  end
  [R,L] = adaptive_single_scale_retinex(X);
  if(size(R)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  [R,L] = adaptive_single_scale_retinex(X,[],[],[],0);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end 
  [R,L] = adaptive_single_scale_retinex(X,10,10,1,0);
  if(size(R)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "adaptive_single_scale_retinex" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "adaptive_single_scale_retinex" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "adaptive_single_scale_retinex" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end



%anisotropic_smoothing
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = anisotropic_smoothing(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = anisotropic_smoothing(X,20);
  if(size(Y)~=size(X))
      ok=0;
  end 

  [R,L] = anisotropic_smoothing(X,[],0);
  if(size(R)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  [R,L] = anisotropic_smoothing(X,10,0);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "anisotropic_smoothing" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "anisotropic_smoothing" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "anisotropic_smoothing" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end








%isotropic_smoothing
ok = 1;
report.test_no = report.test_no + 1;
try
 Y = isotropic_smoothing(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = isotropic_smoothing(X,20);
  if(size(Y)~=size(X))
      ok=0;
  end 

  [R,L] = isotropic_smoothing(X,[],0);
  if(size(R)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  [R,L] = isotropic_smoothing(X,10,0);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "isotropic_smoothing" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "isotropic_smoothing" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "isotropic_smoothing" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end





%multi_scale_retinex
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=multi_scale_retinex(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=multi_scale_retinex(X,[7 15 21]);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  Y=multi_scale_retinex(X,[15]);
  if(size(Y)~=size(X))
      ok=0;
  end 
  Y=multi_scale_retinex(X,[],0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  Y=multi_scale_retinex(X,[7 15 21],0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "multi_scale_retinex" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "multi_scale_retinex" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "multi_scale_retinex" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end




%single_scale_self_quotient_image
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=single_scale_self_quotient_image(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=single_scale_self_quotient_image(X,11);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  Y=single_scale_self_quotient_image(X,7,0.5);
  if(size(Y)~=size(X))
      ok=0;
  end 
  [Y,L]=single_scale_self_quotient_image(X,[],[],0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  
  [Y,L]=single_scale_self_quotient_image(X,7,0.5,0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  if(size(L)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "single_scale_self_quotient_image" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "single_scale_self_quotient_image" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "single_scale_self_quotient_image" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end




%multi_scale_self_quotient_image
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=multi_scale_self_quotient_image(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=multi_scale_self_quotient_image(X,[3 15 29]);
  if(size(Y)~=size(X))
      ok=0;
  end
  
  Y=multi_scale_self_quotient_image(X,[3 5 9 13 29], [0.5 1 2 0.2 4]);
  if(size(Y)~=size(X))
      ok=0;
  end 
  Y=multi_scale_self_quotient_image(X,[3 5 9 13 29], [0.5 1 2 0.2 4], 0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "multi_scale_self_quotient_image" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "multi_scale_self_quotient_image" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "multi_scale_self_quotient_image" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end




%nl_means_normalization
ok = 1;
report.test_no = report.test_no + 1;
try
 Y = nl_means_normalization(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = nl_means_normalization(X,30);
  if(size(Y)~=size(X))
      ok=0;
  end 

  Y = nl_means_normalization(X,80,3);
  if(size(Y)~=size(X))
      ok=0;
  end 
 
  
  [R,L] = nl_means_normalization(X,[],[],0);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end 
  
  [R,L] = nl_means_normalization(X,80,3,0);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "nl_means_normalization" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "nl_means_normalization" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "nl_means_normalization" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end




%steerable_gaussians
ok = 1;
report.test_no = report.test_no + 1;
try
  Y = steerable_gaussians(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y = steerable_gaussians(X,[0.1,1,3]);
  if(size(Y)~=size(X))
      ok=0;
  end 

  Y = steerable_gaussians(X,[0.5,2],6);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  Y = steerable_gaussians(X,[0.1,1,3],[],0);
  if(size(Y)~=size(X))
      ok=0;
  end 

  Y = steerable_gaussians(X,[0.5,2],6,0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "steerable_gaussians" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "steerable_gaussians" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "steerable_gaussians" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end



%wavelet_denoising
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=wavelet_denoising(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=wavelet_denoising(X,'haar');
  if(size(Y)~=size(X))
      ok=0;
  end 

  Y=wavelet_denoising(X,'db1',5);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  [R,L]=wavelet_denoising(X,'haar',2,0);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end

  [R,L]=wavelet_denoising(X,[],[],0);
  if(size(R)~=size(X))
      ok=0;
  end
  if(size(L)~=size(X))
      ok=0;
  end
  
  if(ok)
    report.msg{report.test_no} = 'Function "wavelet_denoising" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "wavelet_denoising" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "wavelet_denoising" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end






%wavelet_normalization
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=wavelet_normalization(X);
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=wavelet_normalization(X,1.3);
  if(size(Y)~=size(X))
      ok=0;
  end 

  Y=wavelet_normalization(X,1.2,'sym1');
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  Y=wavelet_normalization(X,1.4,'haar','sym');
  if(size(Y)~=size(X))
      ok=0;
  end 

  Y=wavelet_normalization(X,[],[],[],0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  Y=wavelet_normalization(X,1.4,'haar','sym',0);
  if(size(Y)~=size(X))
      ok=0;
  end 
  
  if(ok)
    report.msg{report.test_no} = 'Function "wavelet_normalization" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "wavelet_normalization" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "wavelet_normalization" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end





%homomorphic
ok = 1;
report.test_no = report.test_no + 1;
try
  Y=normalize8(homomorphic(X));
  if(size(Y)~=size(X))
      ok=0;
  end
  Y=normalize8(homomorphic(X,2, .25, 2, 0, 5));
  if(size(Y)~=size(X))
      ok=0;
  end 

  if(ok)
    report.msg{report.test_no} = 'Function "homomorphic" is working properly.';
    report.ok(report.test_no)  = 1;
  else
    report.msg{report.test_no} = 'Function "homomorphic" is NOT working properly.';
    report.ok(report.test_no)  = 0;    
  end
catch
  report.msg{report.test_no} = 'Function "homomorphic" is NOT working properly.';
  report.ok(report.test_no)  = 0;  
end


disp('Done.')


%% Report
disp(' ')
disp(' ')
disp('|========================================================================================|')
disp(' ')
disp('VALIDATION REPORT:')
disp(' ')
for i=1:length(report.ok)
    disp(report.msg{i}) 
end
disp(' ')
disp('|========================================================================================|')
disp(' ')
disp('SUMMARY:')
disp(' ')
if sum(report.ok == 0) > 0 
    disp('The following functions reported some errors in their execution. If the "nl_means" ');
    disp('functions are among them, compiling the C/C++ code has failed.')
    disp(' ')
    for i=1:length(report.ok)
        if(report.ok(i) == 0)
            disp(report.msg{i}) 
        end
    end
else
    disp('All functions from the toolbox are working ok.') 
end
disp(' ')
disp('|========================================================================================|')






























