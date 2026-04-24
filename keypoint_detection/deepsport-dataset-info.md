About Dataset
Context

This basketball dataset was acquired under the Walloon region project DeepSport, using the Keemotion system installed in multiple arenas.
We would like to thanks both Keemotion for letting us use their system for raw image acquisition during live productions, and the LNB for the rights on their images.
Content

The dataset is composed of images file (in '.png' format), keypoint ground-truth files (in '.json' format) and masks ground-truth files (in '.png' format).

Images: The dataset is composed of pairs of successive images captured in different basketball arenas during professional games. The cameras capture a fixed part of the basketball court and have a resolution between 2Mpx and 5Mpx. The resulting images have a definition varying between 65px/m (furthest point on court in the arena with the lowest resolution cameras) and 265px/m (closest point on court in the arena with the highest resolution cameras). The delay between the two successive images is 40ms.

Annotations:

Images are provided with camera calibration data where, by convention, the origin (0,0,0) is located on the furthest left corner of the court, x-direction is along the court length, y-direction is along the court width, z-direction is pointing down and distances are expressed in centimetres. Keypoints Ground-Truth (given in the 3D space) and calibration data are stored in json files as a list of annotations with multiple attributes.

Ball keypoint:

    type: "Ball"
    camera: The camera index on which the ball has been annotated (starting at 0)
    position: The (x,y,z) position of the center of the ball in the image
    visible: A boolean stating if at least half of the ball surface is visible

Human poses:

    type: "Player"
    camera: The camera index on which the human has been annotated (starting at 0)
    head: The (x,y,z) position of the center of the head, with z at 1m80 above the court by convention.
    hips: The (x,y,z) position of the center of the hips, with z at 90cm above the court by convention.
    foot1: The (x,y,z) position of one foot (no left/right consideration), with z=0 by convention.
    foot2: The (x,y,z) position of the other foot (no left/right consideration), with z=0 by convention.

Note on Object Keypoint Similarity: To compute OKS, the standard deviation of the unnormalized Gaussian trough which the euclidian distance between predicted and annotated keypoints is passed should be sκi, where s is the object scale and κi is a per-keypont constant that controls falloff. We measured the following values:

    ball: κ = 0.06
    head: κ = 0.12
    hips: κ = 0.16
    feet: κ = 0.08

Segmentation Ground-Truth Mask segmentation is stored in a png file where each pixel correspond to one single instance. The class is given by the 4th digit and the instance ID (starting at 1) is given by the 1st digit:

    Unlabelled class: 0
    Human class: 1
    Ball class: 3

Acknowledgements

This basketball dataset was acquired under the Walloon region project DeepSport, using the Keemotion system installed in multiple arenas.

We would like to thanks both Keemotion for letting us use their system for raw image acquisition during live productions, and the LNB for the rights on their images.
Inspiration

We kindly ask you to mention related publications of the ISPGroup (see bottom of the page) when using this dataset (in publications, video demonstrations…). 
