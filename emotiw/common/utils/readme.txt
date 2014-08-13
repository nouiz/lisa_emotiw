This folder contains 2 keypoints detection models:

- A web service (API) of the face++ implementation:
website: http://www.faceplusplus.com/
The API can detect 83 keypoints (inner point + contour) or 51 (inner points),
so they added some keypoints not initially in the paper (63 keypoints).

You need to have a face++ account to use the API (free to register, a Lisa account already exist) and to create an application on the website.
On the application page, you can see the API KEY and API SECRET, you need to add them on the apikey.cfg file.
You can now use the face++ keypoint detection system.
The facepp.py file is from https://github.com/FacePlusPlus/facepp-python-sdk.

Reference:
Extensive Facial Landmark Localization with Coarse-to-fine Convolutional Neural Network
Erjin Zhou, Haoqiang Fan, Zhimin Cao, Yuning Jiang and Qi Yin
ICCV workshop on 300 Faces in-the-Wild Challenge, 2013.


- Windows executables:
website: http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
The executables need Windows to run (doesn't work with Wine 1.6.2).
Before launching the script you need to verify that Python is authorized through the Windows firewall.
You also need to customize the set_image_path method to your needs (the /data/lisa folder is accessible with a network drive mounted at Q: on Burns)

A script called deepconvcasc.py exposes its methods with RPC through the network thanks to Pyro4 (https://pypi.python.org/pypi/Pyro4).
To expose the DeepConvCascade object to the network you need to run a Pyro4 name server (any machine should work, depending on the network).
You then have to set the PYRO_HOST variable to the Windows machine IP and you will be able to launch the deepconvcasc.py script.

Procedure:
- Verify firewall and the set_image_path method
- start the name server: python -m Pyro4.naming
- set the PYRO_HOST variable: set PYRO_HOST=<HOSTIP>
- launch the script: python deepconvcasc.py

Reference:
Y. Sun, X. Wang, and X. Tang. Deep Convolutional Network Cascade for Facial Point Detection.
In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
