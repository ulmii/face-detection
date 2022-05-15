******************************************************
ChokePoint Dataset
Copyright (C) 2011 NICTA

For details see:

http://arma.sourceforge.net/chokepoint/

and

Y. Wong, S. Chen, S. Mau, C. Sanderson, B.C. Lovell.
Patch-based Probabilistic Image Quality Assessment for Face Selection and Improved Video-based Face Recognition.
IEEE Biometrics Workshop, Computer Vision and Pattern Recognition (CVPR) Workshops, pages 81-88, 2011.
http://doi.org/10.1109/CVPRW.2011.5981881


******************************************************

Recording Environment
P1E : indoor
P1L : indoor
P2E : outdiir
P2L : indoor


Recording Time
P1E*   & P1L*   : 16th April 2010
P2E*.1 & P2L*.1 : 14th May   2010
P2E*.2 & P2L*.2 : 31st May   2010


Camera Details
Camera    : AXIS P1343
Frame Rate: 30 fps
Resolution: 800*600 pixels


Sequence Name
Sequence  : P#*_S#_C#   <Portal.no.type_Sequence.no_Camera.no>
                        < type == E : enter >
                        <         L : leave >


Each folder contains:
all_file.txt        --> list of all images
bg_img.txt          --> list of background images


Sequence with the most frontal view:
P1E_S1_C1
P1E_S2_C2
P1E_S3_C3
P1E_S4_C1
P1L_S1_C1
P1L_S2_C2
P1L_S3_C3
P1L_S4_C1
P2E_S1_C3
P2E_S2_C2
P2E_S3_C1
P2E_S4_C2
P2L_S1_C1
P2L_S2_C2
P2L_S3_C3
P2L_S4_C2


Image optimization method:
jpegtran -progressive -optimize 1.jpg > 1.jpg.b
