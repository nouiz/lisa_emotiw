% A demo script how to combine different functions from the toolbox into a novel technique
% 
% GENERAL DESCRIPTION
% The script shows how different functions from the toolbox can be combined
% to create an new photometric normalization technique.
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
disp(sprintf('This is a Demo script for the INface toolbox. It demonstrates how different functions from the \ntoolbox can be combined to create a powerful normalization technique. Note that this is only an \nexample, any reasonble combination of techniques can be used for the normalization process.\n'))
X=imread('sample_image.bmp');
X=normalize8(imresize(X,[128,128],'bilinear'));

%% Prepare figure - reflectance
figure(1)
subplot(3,2,1)
imshow(normalize8(X),[])
hold on 
axis image
axis off
title('Original')


%% Build a novel technique
disp('Remap histogram to resamble Lognormal distribution')
Y=fitt_distribution(X,2,[0,0.25]);
figure(1)
subplot(3,2,2)
imshow(normalize8(Y),[])
title('Step 1: LogNorm distribution')
disp('Done.')

%% Compute estimate of luminance
disp('Compute estimate of luminance function')
[R,L] = anisotropic_smoothing_stable(Y,15,0);
figure(1)
subplot(3,2,3)
imshow(normalize8(L),[])
title('Step 2: Luminance estimate')
disp('Done.')

%% do threshold filtering
disp('Corret luminance using threshold filtering')
L1=threshold_filtering(L, 9, 2);
figure(1)
subplot(3,2,4)
imshow(normalize8(L1),[])
title('Step 3: Corrected luminance')
disp('Done.')

%% compute new reflectance
disp('Compute reflectance with corrected luminance')
R = Y - L1;
figure(1)
subplot(3,2,5)
imshow(normalize8(R),[])
title('Step 4: Compute reflectance')
disp('Done.')

%% do robust postprocessing
disp('Do some postprocessing')
[R, dummy] =histtruncate(R, 0.2, 0.2);
figure(1)
subplot(3,2,6)
imshow(normalize8(R),[])
title('Step 5: Process reflectance')
disp('Done.')






















