% The function performs threshold filtering of an image
% 
% PROTOTYPE
%       Y=threshold_filtering(X, mask_size, percentage);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y=threshold_filtering(X, 9, 1);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y=threshold_filtering(X, [], 5);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y=threshold_filtering(X);
%       figure,imshow(X);
%       figure,imshow(Y,[]);
%
% GENERAL DESCRIPTION
% The function performs threshold filtering as suggested in the reference
% paper. Basically, the technique applies a local averaging filter to a
% local pixel neighboorhood of size "mask_size X mask_size", but only if 
% the center pixel value is larger than a threshold. This threshold is 
% determined adaptively in such a way, that only the specified "percentage" 
% of pixels have a larger value than the threshold. Efectively, this means
% that only the specified "percentage" of picels are replaced through the 
% averaging operation.  
% 
% REFERENCES
% This function is an implementation of the threshold filtering approach
% proposed in:
%
% X. Xie, W.S. Zheng, J. Lai, P.C. Yuen, C.Y. Suen: Normalization of Face
% illumination Based on Large- and Small- Scale Features, IEEE Transactions
% on Image Processing, vol. 20, no. 7, 2011, pp. 1807-1821.
% 
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% mask_size             - a scalar value (an odd number) determining the 
%                         size of the local pixel neigboorhood, on which the 
%                         threshold averging is perfomed; default = 9 
% percentage            - a scalar value determining the percentage of
%                         pixels that need to be larger than the adaptive
%                         threshold
%
% OUTPUTS:
% Y                     - a tresholf filtered grey-scale image 
% 
%
% NOTES / COMMENTS
% This function is needed by a few other functions actually performing
% photometric normalization. It applies threshold filtering to an input 
% image.
%
% The function was tested with Matlab ver. 7.11.0.584 (R2010b).
% 
% 
% RELATED FUNCTIONS (SEE ALSO)
% histtruncate  - a function provided by Peter Kovesi
% normalize8    - auxilary function
% 
% 
% ABOUT
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

function Y=threshold_filtering(X, mask_size, percentage);

%% default return value
Y=[];

%% Parameter check
if nargin==1
    mask_size = 9;
    percentage= 1;
elseif nargin == 2
    if isempty(mask_size)
        mask_size = 9;
    end
    percentage = 1;
elseif nargin == 3
    if isempty(mask_size)
        mask_size = 9;
    end
    if isempty(percentage)
        percentage = 1;
    end
    
else
   disp('Error: The function takes at most three arguments.');
   return;
end

if(mod(mask_size,2)~=1)
    disp('Error: "mask_size" needs to be an odd number.');
    return; 
end

%% Init. operations
X=double(X);
[a,b]=size(X);
pix_num = a*b;

%calculate needed padding
in_one_dim = (mask_size-1)/2;

%pad image
XP = padarray(X,[in_one_dim,in_one_dim],'symmetric','both');


%% Threshold calculation
max_val = max(max(X));
min_val = min(min(X));

delta = (max_val-min_val)/5000;
histo = zeros(1,5001);
cum_perc = 100-percentage;
curr_threshold = -1;
for threshold = min_val:delta:max_val
    if((sum(sum(X>threshold))/pix_num)*100 < percentage)
        curr_threshold = threshold-delta;
        break;
    end
end


%% the actual filtering
Y=zeros(a,b);

% main loop
for i=in_one_dim+1:a+in_one_dim
    for j=in_one_dim+1:b+in_one_dim
        if XP(i,j)<=curr_threshold
            Y(i-in_one_dim, j-in_one_dim) = XP(i,j);
        else
            Y(i-in_one_dim, j-in_one_dim) = mean(mean(XP(i-in_one_dim:i+in_one_dim,j-in_one_dim:j+in_one_dim)));
        end
    end
end


















