% The function applies the adaptive non-local means normalization technique to an image
% 
% PROTOTYPE
% [R,L] = adaptive_nl_means_normalization(X,h,N,normalize);
% 
% USAGE EXAMPLE(S)
% 
%     Example 1:
%       X=imread('sample_image.bmp');
%       Y = adaptive_nl_means_normalization(X);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 2:
%       X=imread('sample_image.bmp');
%       Y = adaptive_nl_means_normalization(X,20);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 3:
%       X=imread('sample_image.bmp');
%       Y = adaptive_nl_means_normalization(X,150,3);
%       figure,imshow(X);
%       figure,imshow((Y),[]);
% 
%     Example 4:
%       X=imread('sample_image.bmp');
%       [R,L] = adaptive_nl_means_normalization(X,150,[],0);
%       figure,imshow(X);
%       figure,imshow((R),[]);
%       figure,imshow((L),[]);
% 
%     Example 5:
%       X=imread('sample_image.bmp');
%       [R,L] = adaptive_nl_means_normalization(X,[],[],0);
%       figure,imshow(X);
%       figure,imshow((R),[]);
%       figure,imshow((L),[]);
% 
%     Example 6:
%       X=imread('sample_image.bmp');
%       [R,L] = adaptive_nl_means_normalization(X,150,3,1);
%       figure,imshow(X);
%       figure,imshow((R),[]);
%       figure,imshow((L),[]);
% 
%
%
% GENERAL DESCRIPTION
% The function performs photometric normalization of the image X using the
% adaptive non-local means algorithm. The algorithm constructs a smoothed image
% based on a weighted sum of similar patches comprising the image. The
% smoothed image is then used to estimate the reflectance which should be
% illumination invariant. In contrast to the original non-local means
% technique this technique varies the decay parameter of exponential 
% function in the non-local means expression to achieve adaptive smoothing. 
% 
% The function is intended for use in face recognition experiments and the
% default parameters are set in such a way that a "good" normalization is
% achieved for images of size 128 x 128 pixels. Of course the term "good" is
% relative. The default parameters are set as used in the chapter of the
% AFIA book.
%
%
% 
% REFERENCES
% This function is an implementation of the adaptive non-local means based 
% photometric normalization proposed in technique described in:
%
% Štruc, V. in Pavešiæ, N.: Illumination Invariant Face Recognition by
% Non-Local Smoothing, Proceedings of BIOID MultiComm, LNCS 5707, Springer, 
% pp. 1-8, September 2009.
% 
%
%
%
% INPUTS:
% X                     - a grey-scale image of arbitrary size
% h                     - a scalar value controling the decay of the
%                         exponential function, in this implementation it
%                         controls the maximum value of the parameter,
%                         i.e., the value to which the maximum value of the
%                         processed contrast image is mapped to
% N                     - a scalar value defining the neigborhood size,
%                         i.e., the size of the patches to be used in the
%                         non-local means algorithm, default value "N=2"
% normalize             - a parameter controlling the post-processing
%                         procedure:
%                            0 - no normalization
%                            1 - perform basic normalization (truncation 
%                                of histograms ends and normalization to the 
%                                8-bit interval) - default
% 
% For a more detailed description of the parameters please type 
% "help perform_nl_means_adap" into Matlabs command prompt. 
% 
%
% OUTPUTS:
% R                     - a grey-scale image processed with the adaptive
%                         non-local means normalization technique (the reflectance)
% L                     - the estimated luminance function 
%                         
%
% NOTES / COMMENTS
% This function applies the adaptive-non-local-means-based normalization to 
% the grey-scale image X. 
%
% The function was tested with Matlab ver. 7.5.0.342 (R2007b) and Matlab 
% ver. 7.11.0.584 (R2010b).
%
% 
% RELATED FUNCTIONS (SEE ALSO)
% histtruncate          - a function provided by Peter Kovesi
% normalize8            - auxilary function
% perform_nl_means_adap - a function based on the code of Gabriel Peyre
% 
% 
% ABOUT
% Created:        26.8.2009
% Last Update:    12.10.2011
% Revision:       2.0
% 
%
% WHEN PUBLISHING A PAPER AS A RESULT OF RESEARCH CONDUCTED BY USING THIS CODE
% OR ANY PART OF IT, PLEASE MAKE A REFERENCE TO THE FOLLOWING PUBLICATIONS:
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

function [R,L] = adaptive_nl_means_normalization(X,h,N,normalize);

%% Init
R=[];
L=[];

%% Parameter checking
if nargin == 1
    h = 80;
    N = 2;
    normalize=1;
elseif nargin == 2  
    if isempty(h)
        h=80;
    end
    N = 2;
    normalize=1;
elseif nargin == 3
    if isempty(h)
        h=80;
    end
    
    if isempty(N)
        N = 2;
    end
    normalize=1;
elseif nargin == 4
    if isempty(h)
        h=80;
    end
    
    if isempty(N)
        N = 2;
    end
    if ~(normalize==1 || normalize==0)
        disp('Error: The fourth parameter can only be 0 or 1.');
        return;
     end
elseif nargin >4
    disp('Error! Wrong number of input parameters.')
    return;
end


%% Init. operations
X=normalize8(X);
options.T=h;
options.k=N;

%% Produce the local contrast image for adaptation
Contr = get_local_contrast(X,h);
options.Tm=Contr;

%% Apply non-local means using Gabriels toolbox
[M1,Wx,Wy,mask_copy] = perform_nl_means_adap(double(X), options);

%% Produce reflectance
R=log(double(X)+1)-log(M1+1);
L=log(M1+1);

%% Do some final post-processing (or not)
if normalize ~= 0
    R = normalize8(histtruncate(R,0.2,0.2));
    L = normalize8(L);
end
   




%% This function produces the local contrast image
function Y = get_local_contrast(X,params);

[a,b]=size(X);
X=normalize8(X);

%padding with same as in X
X1=zeros(a,b+1);
X1(1:a,2:b+1)=X;
X1(1:a,1)=X(:,1);

X2=zeros(a,b+1);
X2(1:a,1:b)=X;
X1(1:a,b+1)=X(:,b);

%contrasts (Michelson) - west and east
Gxl = abs(X1(:,2:b+1)-X2(:,2:b+1))./(abs(X1(:,2:b+1)+X2(:,2:b+1))+0.1);
Gxr = abs(X1(:,1:b)-X2(:,1:b))./(abs(X1(:,1:b)+X2(:,1:b))+0.1);


%pading with same as X
X1=zeros(a+1,b);
X1(2:a+1,1:b)=X;
X1(1,:)=X(1,:);

X2=zeros(a+1,b);
X2(1:a,1:b)=X;
X2(a+1,:)=X(a,:);

%contrasts (Michelson) - north and west
Gyu = abs(X1(2:a+1,:)-X2(2:a+1,:))./(abs(X1(2:a+1,:)+X2(2:a+1,:))+0.1);
Gyd = abs(X1(1:a,:)-X2(1:a,:))./(abs(X1(1:a,:)+X2(1:a,:))+0.1);

%mean contrast image 
Y=Gxl+Gxr+Gyu+Gyd;


%remap the contrast so it is invertible
Y=double(normalize_to_1_255_interval((Y)));

%conduction functions - we use last
% Y=1./(1+sqrt(Y));
% Y=1./(1+(Y).^2);
Y=1./(1+(Y));

%remap again to [0 255]
Y=normalize8(Y);

%perform threshold filtering to correct for holes and take logarithm
Y=thresh_filt(Y,200,3);
Y=log(Y+1);

%we map back to [1,255]
Y=double(normalize_to_1_255_interval((Y)));

%finally we adjust the range as defined in params (the upper value is needed - h_max)
Y=normalize_to_1_100_interval(Y,params);



%% This function maps an image into the range of 1-255
function Y=normalize_to_1_255_interval(X);


X=double(X);
[a,b]=size(X);
max_v_x = max(max(X));
min_v_x = min(min(X));

Y=ceil(((X - min_v_x*ones(a,b))./(max_v_x*(ones(a,b))-min_v_x*(ones(a,b))))*254)+1;

%% This function maps an image into the range of 0.01-params
function Y=normalize_to_1_100_interval(X,params);

X=double(X);
[a,b]=size(X);
max_v_x = max(max(X));
min_v_x = min(min(X));

Y=(((X - min_v_x*ones(a,b))./(max_v_x*(ones(a,b))-min_v_x*(ones(a,b))))*params)+0.01;

%% This function performs threshold filtering
function X=thresh_filt(Y,thresh,iter);

[a,b]=size(Y);

X=Y;
Y1=zeros(a+2,b+2);
Y1(2:a+1,2:b+1)=Y;
Y1(1,2:b+1)=Y(1,:);
Y1(a+1,2:b+1)=Y(a,:);
Y1(2:a+1,b+1)=Y(:,b);
Y1(2:a+1,1)=Y(:,1);
Y1(1,1)=Y(1,1);
Y1(a+2,b+2)=Y(a,b);
Y1(1,b+2)=Y(1,b);
Y1(a+2,1)=Y(a,1);

for m=1:iter
    for i=2:a+1
        for j=2:b+1
            if Y1(i,j)>thresh
                tmp=Y1(i-1:i+1,j-1:j+1);
                X(i-1,j-1)=median(tmp(:));
            end
        end
    end
    Y1(2:a+1,2:b+1)=X;
end

