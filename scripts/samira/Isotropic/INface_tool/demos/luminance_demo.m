% A demo script showing the procedure of computing the luminance functions.
% 
% GENERAL DESCRIPTION
% The script applies a selected set of photometric normalization techniques
% to the input image and displays the estimated luminance functions. Here,
% not all techniques are shown, since not all of them estimete the
% luminance function.
% 
% 
%
% NOTES / COMMENTS
% The script was tested with Matlab ver. 7.5.0.342 (R2007b) and WindowsXP 
% as well as Matlab ver. 7.11.0.584 (R2010b) running on Windows 7.
% 
% ABOUT
% Created:        28.8.2009
% Last Update:    13.10.2011
% Revision:       2.0
% 
%
% WHEN PUBLISHING A PAPER AS A RESULT OF RESEARCH CONDUCTED BY USING THIS CODE
% OR ANY PART OF IT, MAKE A REFERENCE TO THE FOLLOWING PUBLICATIONS:
%
% 1. Štruc V., Pavešiæ, N.:Photometric normalization techniques for illumination 
% invariance, in: Y.J. Zhang (Ed), Advances in Face Image Analysis: Techniques 
% and Technologies, IGI Global, pp. 279-300, 2011.
% (BibTex available from: http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/IGI.bib)
% 
% 2. Štruc, V., Pavešiæ, N.: Gabor-based kernel-partial-least-squares 
% discrimination features for face recognition, Informatica (Vilnius), 
% vol. 20, no. 1, pp. 115-138, 2009.
% (BibTex available from: http://luks.fe.uni-lj.si/sl/osebje/vitomir/pub/InforVI.bib)
% 
%
% Official website:
% If you have down-loaded the toolbox from any other location than the
% official website, plese check the following link to make sure that you
% have the most recent version:
% 
% http://luks.fe.uni-lj.si/sl/osebje/vitomir/face_tools/INFace/index.html
% 
% 
% Copyright (c) 2011 Vitomir Štruc
% Faculty of Electrical Engineering,
% University of Ljubljana, Slovenia
% http://luks.fe.uni-lj.si/en/staff/vitomir/index.html
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files, to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
% 
% October 2011


%% Load sample image
disp(sprintf('This is a Demo script for the INface toolbox. It demonstrates the computation of the\nluminance function of the input image. Since not all techniques produce an estimate \nof the luminance function, only a subset of function is used here.\n'))
X=imread('sample_image.bmp');
X=normalize8(imresize(X,[128,128],'bilinear'));

%% Prepare figure - reflectance
figure(1)
subplot(3,4,1)
imshow(normalize8(X),[])
hold on 
axis image
axis off
title('Original')


%% Apply the photometric normalization techniques

%SSR
disp('Applying the single scale retinex technique.')
[Y,L]=single_scale_retinex(X); %reflectance

%display the images
figure(1)
subplot(3,4,2)
imshow(normalize8(L),[])
title('SSR')
disp('Done.')

%ASR
disp('Applying the adaptive single scale retinex technique.')
siz = [3 5 11 15];
sigma = [0.9 0.9 0.9 0.9];
[Y,L]=adaptive_single_scale_retinex(X); %reflectance

%display the images
figure(1)
subplot(3,4,3)
imshow(normalize8(L),[])
title('ASR')
disp('Done.')


%SSQ
disp('Applying the single scale self quotient image technique.')
[Y,L]=single_scale_self_quotient_image(X); %reflectance

%display the images
figure(1)
subplot(3,4,4)
imshow(normalize8(L),[])
title('SSQ')
disp('Done.')


%WD
disp('Applying the wavelet-denoising-based normalization technique.')
[Y,L]=wavelet_denoising(X,'coif1',3); %reflectance

%display the images
figure(1)
subplot(3,4,5)
imshow(normalize8(L),[])
title('WD')
disp('Done.')


%IS
disp('Applying the isotropic diffusion-based normalization technique.')
[Y,L]=isotropic_smoothing(X); %reflectance


%display the images
figure(1)
subplot(3,4,6)
imshow(normalize8(L),[])
title('IS')
disp('Done.')


%AS
disp('Applying the anisotropic diffusion-based normalization technique.')
[Y,L]=anisotropic_smoothing(X); %reflectance

%display the images
figure(1)
subplot(3,4,7)
imshow(normalize8(L),[])
title('AS')
disp('Done.')


%NLM
disp('Applying the non-local-means-based normalization technique.')
[Y,L]=nl_means_normalization(X); %reflectance

%display the images
figure(1)
subplot(3,4,8)
imshow(normalize8(L),[])
title('NLM')
disp('Done.')


%ANL
disp('Applying the adaptive non-local-means-based normalization technique.')
[Y,L]=adaptive_nl_means_normalization(X); %reflectance


%display the images
figure(1)
subplot(3,4,9)
imshow(normalize8(L),[])
title('ANL')
disp('Done.')


%MAS
disp('Applying the modified anisotropc smoothing normalization technique.')
[Y,L]= anisotropic_smoothing_stable(X); %reflectance


%display the images
figure(1)
subplot(3,4,10)
imshow(normalize8(L),[])
title('MAS')
disp('Done.')





















