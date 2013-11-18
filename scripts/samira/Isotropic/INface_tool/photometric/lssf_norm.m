% The function applies the large and small scale features appoach to an input image.
% 
% PROTOTYPE
% Y = lssf_norm(X, normalize)
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = lssf_norm(X);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = lssf_norm(X, 0);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the 
% large and small scale features approach from the reference paper. Hre,
% the first input argument is image to be normalized, and the second is
% again a parameter controlig whether the processed image is scaled to the
% 8-bit interval in the end or not.
% 
% Different from the original apporach, we have not implemented the LTV
% model for the decomposition in the first step. Rather we have used the
% single scale retinex technique for both steps - normalization of the
% small scale as well as normalization of the large scale features. One can
% argue that this is not as effective as the method proposed by the
% authors, esspecially considering the fact that the authors show
% differences in the (first-spet) decomposition, when different techniques
% are used (Fig. 3). However, it is obvious that the same effect can be 
% achieved by tuning the methods parameters and basically adjusting the 
% cut-off frequency, since most photometric normalization technique act as 
% sort of highpass filters. Performing a second normalization step on the 
% large scale features has (approximatelly) the effect of adjusting the cut-off
% frequency. Hence, we used again the SSR technique for the second step.
% 
% The only possiblity to improve upon the results, is to use some
% background knowledge - the PCA basis as suggested by the authors.
% However, this step makes the techniques limited to frontal faces or to be
% general requires a pose identification procedure and additinal subspaces.
% We have therefore not implemented the technique with the NPL approach.
% 
% The last step (the preimaging) is actually a recognition step and more or
% less useless for recognition, but you have to do that by hand.
% 
% Based on the above comments I would use this function with care, since it
% is basicaly the same as SSR with adjusted parameters. You are, however,
% welcomed to replace the SSR technique with some other method from this
% toolbox. 
% 
% A useful technique from the reference paper is the treshold averging,
% which is also implemented in this toolbox. 
% 
% 
% REFERENCES
% This function is an implementation of the large ans small scale features
% (SSLF) technique proposed in:
%
% X. Xie, W.S. Zheng, J. Lai, P.C. Yuen, C.Y. Suen: Normalization of Face
% illumination Based on Large- and Small- Scale Features, IEEE Transactions
% on Image Processing, vol. 20, no. 7, 2011, pp. 1807-1821.
% 
% 
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (range 
%                                adjustment to the 8-bit interval) - default
%
% OUTPUTS:
% Y                     - a grey-scale image normalized using the lssf
%                         technique
% 
%
% NOTES / COMMENTS
% This function applies the LSSF appraoch to the grey-scale image X. 
% 
%
% 
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% normalize8            - auxilary function
% 
% Created:        19.8.2009
% Last Update:    11.10.2011
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
function Y = lssf_norm(X, normalize)

%% Default results
Y=[];

%% parameter checking
if nargin == 1
    normalize=1;
elseif nargin == 2
    if isempty(normalize)
        normalize = 1;
    end
elseif nargin > 2
   disp('Error: the function takes at most two parameters.');
   return;
end



%% Init. operations
[a,b]=size(X);
X1=normalize8(X); 


%% First step - decomposition using provided method
[R,L] = single_scale_retinex(X1,5, 0);
R=exp(R);
L=exp(L);


if (sum(size(R)~= size(X)) || sum(size(L)~= size(X)))
   disp('The method specified by the argument "method" did not return the desired result. Terminating');
   return;
end

%% Second step - threshold filtering on small-scale features
R1=threshold_filtering(R, 5, 1);

%% Third step - processing of small scale features
[Rs,Ls] = single_scale_retinex(L,2, 0);
Y = R1.*exp(Rs);

%% postprocessing
if normalize ~= 0
    Y=normalize8(Y);  
end
 















