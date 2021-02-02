from Colorizer import *

color = Colorizer(use_cuda=False)

# color.processImg('Input/6.jpg')


# vidList = ['video.avi','video1.avi','video2.avi','video3.avi','video4.avi']
# for vid in vidList:
#     color.processVideo('Input/'+vid)

color.processVideo('Input/video4.avi')