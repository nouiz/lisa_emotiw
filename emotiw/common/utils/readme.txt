This folder contains 2 keypoints detection models:

- A web service (API) of the Face++ implementation:
website: http://www.faceplusplus.com/
The API can detect 83 keypoints (inner point + contour) or 51 (inner points),
so they added some keypoints not initially in the paper (63 keypoints).

You need to have a Face++ account to use the API (free to register, a Lisa account already exist) and to create an application on the website.
On the application page, you can see the API KEY and API SECRET, you need to add them on the apicfg.py file.
You can now use the Face++ keypoint detection system.
The facepp.py file is from https://github.com/FacePlusPlus/facepp-python-sdk.

Reference:
Extensive Facial Landmark Localization with Coarse-to-fine Convolutional Neural Network
Erjin Zhou, Haoqiang Fan, Zhimin Cao, Yuning Jiang and Qi Yin
ICCV workshop on 300 Faces in-the-Wild Challenge, 2013.


- Windows executables:
website: http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
The executables need Windows to run (doesn't work with Wine 1.6.2).
A script called deepconvcasc.py (available in deepconvcascade) exposes its methods with RPC through the network thanks to Pyro4 (https://pypi.python.org/pypi/Pyro4).

Before running the script you need to verify that Python is authorized through the Windows firewall
and you have to set the PYRO_HOST environment variable to the Windows machine IP:
set PYRO_HOST=<HOSTIP>

You can change the name of the Pyro4 object with the register method of the daemon object
and you also can change on which port the object listen to when initializing the daemon.
You also need to customize the different paths in deepconvcasc.py to your needs.

After running the script, it prints the Pyro4 URI of the object with the name/host/port you configured.
You then only need to copy/paste the URI in apicfg.py and you are good to go.

Procedure:
- Verify firewall
- Set the PYRO_HOST variable: set PYRO_HOST=<HOSTIP>
- Run the script: python deepconvcasc.py
- Look at the printed URI and put it in the apicfg.py file

Reference:
Y. Sun, X. Wang, and X. Tang. Deep Convolutional Network Cascade for Facial Point Detection.
In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
