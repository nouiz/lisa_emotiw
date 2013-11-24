% A demo script showing the application of the photometric normalization techniques on a sample image.
% 
% GENERAL DESCRIPTION
% The script applies all normalization techniques of the INface toolbox to 
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
disp(sprintf('This is a Demo script for the INface toolbox. It applies all of the \nphotometric normalization techniques contained in the toolbox to a sample \nimage and displays the estimated reflectance (illumination invariant) part \nof the image in a figure.\n'))
X=imread('sample_image.bmp');
X=normalize8(imresize(X,[128,128],'bilinear'));

%% Prepare figure - reflectance
figure(1)
subplot(6,4,1)
imshow(normalize8(X),[])
hold on 
axis image
axis off
title('Original')


%% Apply the photometric normalization techniques

%SSR
disp('Applying the single scale retinex technique.')
Y=single_scale_retinex(X); %reflectance

%display the images
figure(1)
subplot(6,4,2)
imshow(normalize8(Y),[])
title('SSR')
disp('Done.')

%MSR
disp('Applying the mutli scale retinex technique.')
Y=multi_scale_retinex(X); %reflectance

%display the images
figure(1)
subplot(6,4,3)
imshow(normalize8(Y),[])
title('MSR')
disp('Done.')


%ASR
disp('Applying the adaptive single scale retinex technique.')
siz = [3 5 11 15];
sigma = [0.9 0.9 0.9 0.9];
Y=adaptive_single_scale_retinex(X); %reflectance

%display the images
figure(1)
subplot(6,4,4)
imshow(normalize8(Y),[])
title('ASR')
disp('Done.')


%HOMO
disp('Applying homomorphic filtering.')
Y=homomorphic(X); %reflectance

%display the images
figure(1)
subplot(6,4,5)
imshow(normalize8(Y),[])
title('HOMO')
disp('Done.')


%SSQ
disp('Applying the single scale self quotient image technique.')
Y=single_scale_self_quotient_image(X); %reflectance

%display the images
figure(1)
subplot(6,4,6)
imshow(normalize8(Y),[])
title('SSQ')
disp('Done.')



%MSQ
disp('Applying the multi scale self quotient image technique.')
Y=multi_scale_self_quotient_image(X); %reflectance

%display the images
figure(1)
subplot(6,4,7)
imshow(normalize8(Y),[])
title('MSQ')
disp('Done.')


%DCT
disp('Applying the DCT-based normalization technique.')
Y=DCT_normalization(X); %reflectance

%display the images
figure(1)
subplot(6,4,8)
imshow(normalize8(Y),[])
title('DCT')
disp('Done.')



%WA
disp('Applying the wavelet-based normalization technique.')
Y=wavelet_normalization(X); %reflectance

%display the images
figure(1)
subplot(6,4,9)
imshow(normalize8(Y),[])
title('WA')
disp('Done.')


%WD
disp('Applying the wavelet-denoising-based normalization technique.')
Y=wavelet_denoising(X,'coif1',3); %reflectance

%display the images
figure(1)
subplot(6,4,10)
imshow(normalize8(Y),[])
title('WD')
disp('Done.')


%IS
disp('Applying the isotropic diffusion-based normalization technique.')
Y=isotropic_smoothing(X); %reflectance

%display the images
figure(1)
subplot(6,4,11)
imshow(normalize8(Y),[])
title('IS')
disp('Done.')


%AS
disp('Applying the anisotropic diffusion-based normalization technique.')
Y=anisotropic_smoothing(X); %reflectance

%display the images
figure(1)
subplot(6,4,12)
imshow(normalize8(Y),[])
title('AS')
disp('Done.')


%NLM
disp('Applying the non-local-means-based normalization technique.')
Y=nl_means_normalization(X); %reflectance

%display the images
figure(1)
subplot(6,4,13)
imshow(normalize8(Y),[])
title('NLM')
disp('Done.')


%ANL
disp('Applying the adaptive non-local-means-based normalization technique.')
Y=adaptive_nl_means_normalization(X); %reflectance

%display the images
figure(1)
subplot(6,4,14)
imshow(normalize8(Y),[])
title('ANL')
disp('Done.')


%SF
disp('Applying the steerable filter based normalization technique.')
Y = steerable_gaussians(X); %reflectance

%display the images
figure(1)
subplot(6,4,15)
imshow(normalize8(Y),[])
title('SF')
disp('Done.')


%MAS
disp('Applying the modified anisotropc smoothing normalization technique.')
Y = anisotropic_smoothing_stable(X); %reflectance

%display the images
figure(1)
subplot(6,4,16)
imshow(normalize8(Y),[])
title('MAS')
disp('Done.')


%Gradientfaces
disp('Applying the Gradientfaces normalization technique.')
Y = gradientfaces(X); %reflectance

%display the images
figure(1)
subplot(6,4,17)
imshow(normalize8(Y),[])
title('GRF')
disp('Done.')


%DOG
disp('Applying a DoG filtering-based normalization technique.')
Y = dog(log(normalize8(X)+1)); %reflectance

%display the images
figure(1)
subplot(6,4,18)
imshow(normalize8(Y),[])
title('DOG')
disp('Done.')


%Tan and Triggs
disp('Applying the Tan and Triggs normalization technique.')
Y = tantriggs(X); %reflectance

%display the images
figure(1)
subplot(6,4,19)
imshow(normalize8(Y),[])
title('TT')
disp('Done.')


%Weberfaces
disp('Applying the single scale Weberfaces normalization technique.')
Y = weberfaces(X); %reflectance

%display the images
figure(1)
subplot(6,4,20)
imshow(normalize8(Y),[])
title('SSW')
disp('Done.')



%multi scale Weberfaces
disp('Applying the multi scale Weberfaces normalization technique.')
Y = multi_scale_weberfaces(X); %reflectance

%display the images
figure(1)
subplot(6,4,21)
imshow(normalize8(Y),[])
title('MSW')
disp('Done.')


%lssf technique
disp('Applying the llsf normalization technique.')
Y = lssf_norm(X); %reflectance

%display the images
figure(1)
subplot(6,4,22)
imshow(normalize8(Y),[])
title('LSSF')
disp('Done.')

























