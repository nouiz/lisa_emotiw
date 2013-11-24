%HISTOGRAMS_DEMO A demo script showing the application of the histogram manipulation techniques on a sample image.
% techniques on a sample image.
% 
% GENERAL DESCRIPTION
% The script applies all histogram manipulation techniques of the INface toolbox to 
% a sample image and displays the results in a figure. 
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
disp(sprintf('This is a Demo script for the INface toolbox. It applies all of the \nhistogram manipulation techniques contained in the toolbox to a sample \nimage and displays the results.\n'))
X=imread('sample_image.bmp');
X=normalize8(imresize(X,[128,128],'bilinear'));

%% Prepare figure - reflectance
figure(1)
subplot(3,4,1)
imshow(normalize8(X),[])
hold on 
axis image
axis off
title('Orig.')

subplot(3,4,2)
hist(normalize8(X(:)),255)
axis([-1 256 0 600])
title('Histogram of Orig')


%% Apply the photometric normalization techniques

%rank normalization
disp('Applying rank normalization.')
Y=rank_normalization(X); %reflectance

%display the images
figure(1)
subplot(3,4,3)
imshow(normalize8(Y),[])
title('HQ')
disp('Done.')

subplot(3,4,4)
hist(normalize8(Y(:)),255)
axis([-1 256 0 600])
title('Histogram of HQ')


%histogram truncating
disp('Applying histogram truncating.')
Y=histtruncate(X,10,10); %reflectance

%display the images
figure(1)
subplot(3,4,5)
imshow(normalize8(Y),[])
title('HT')
disp('Done.')

subplot(3,4,6)
hist(normalize8(Y(:)),255)
axis([-1 256 0 600])
title('Histogram of HT')



%histogram remapping - normal
disp('Mapping normal distribution.')
Y=fitt_distribution(X,1,[0,1]); %reflectance

%display the images
figure(1)
subplot(3,4,7)
imshow(normalize8(Y),[])
title('ND')
disp('Done.')

subplot(3,4,8)
hist(normalize8(Y(:)),255)
axis([-1 256 0 600])
title('Histogram of ND')



%histogram remapping - lognormal
disp('Mapping lognormal distribution.')
Y=fitt_distribution(X,2,[0,0.25]); %reflectance

%display the images
figure(1)
subplot(3,4,9)
imshow(normalize8(Y),[])
title('LN')
disp('Done.')

subplot(3,4,10)
hist(normalize8(Y(:)),255)
axis([-1 256 0 600])
title('Histogram of LN')



%histogram remapping - exponential
disp('Mapping exponential distribution.')
Y=fitt_distribution(X,3,[0.01]); %reflectance

%display the images
figure(1)
subplot(3,4,11)
imshow(normalize8(Y),[])
title('EX')
disp('Done.')

subplot(3,4,12)
hist(normalize8(Y(:)),255)
axis([-1 256 0 600])
title('Histogram of EX')


%% Print out legend

disp(sprintf('\nLEGEND:\n'))
disp(sprintf('Orig. - original image'))
disp(sprintf('HQ - histogram equalized image'))
disp(sprintf('HT - histogram truncated image'))
disp(sprintf('NL - normal distribution mapped to the histogram of the image'))
disp(sprintf('LN - log-normal distribution mapped to the histogram of the image'))
disp(sprintf('EX - exponential distribution mapped to the histogram of the image'))






















