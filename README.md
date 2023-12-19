# Road2Sat: CV-2023 Project repository
## Group members
|Name | Contact |
|---|---|
|Rizwan Ahmad Bhatti | 21030005@lums.edu.pk |
|Faizan Owais | 24100119@lums.edu.pk |

## Directory structure
```
Road2Sat/
├─ dataset/
├─ gen/
├─ resources/
├─ resources/
Road2Sat.py
README.md
requirements.txt
```
### Road2Sat.py
Main script that calculates running homography and warps images in dataset.

### requirements.txt
Consists of necessary python libraries and dependencies.

### README.MD
This file.

### resources/scripts/roadSegmentation.py
Utility script to create dataset with drivable area extracted and objects encoded in json file.

### resources/scripts/selectedPointsHomography.py
Takes input for points on the road reference image to be projected to satellite reference image.

### resources/scripts/video2frames.py
Utility script to generate images dataset by extracting frames from input video.

### resources/models/YOLOP
Submodule for road segmentation and object detection

### resources/models/Stitch-images-using-SuperGlue-GNN
Submodule for corresponding points calculation using superglue

### readme.md
This file.

### results
Results are found in gen folder. mosaic.jpg is a generated file along with cpoint_frames and p_frames.
