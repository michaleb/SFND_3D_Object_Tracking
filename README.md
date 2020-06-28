# SFND 3D Object Tracking

In this project I use keypoint detectors, descriptors, and methods to match them between successive images. Detect objects in an image using the YOLO deep-learning framework and associate regions in a camera image with Lidar points in 3D space. 

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Current local configuration
* cmake = 3.13.2
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make = 4.1 (Linux - Ubuntu 18.04.4 LTS)
  * Linux: make is installed by default on most Linux distros
* OpenCV = 4.3.0 (N.B. cv::SIFT::create() to implement this detector type)
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.3.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.3.0)
* gcc/g++ = 7.5.0
  * Linux: gcc / g++ is installed by default on most Linux distros

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

[//]: # (Image References)

[image0]: /images/ttc-estimate_shitomasi-brisk.png "TTC Camera estimate -- detector/descriptor"
[image1]: /images/ttc-estimate_fast-brief.png "TTC Camera estimate -- detector/descriptor"
[image2]: /images/ttc-estimate_sift-brief.png "TTC Camera estimate -- detector/descriptor"
[image3]: /images/ttc-estimate_sift-brisk.png "Best TTC Camera estimate -- detector/descriptor"
[image4]: /images/ttc-estimate_sift-sift.png "TTC Camera estimate -- detector/descriptor"
[image5]: /images/imgIndex4.png "Median distance closer to Xmin"
[image6]: /images/imgIndex5.png "Median distance closer to 8m marker"
[image7]: /images/imgIndex11.png "Median lower - distances less distributed"
[image8]: /images/imgIndex12.png "Median higher - distances more distributed"


## Implementing the code

### 1. Match 3D Objects
I used a quicker approach for the matching bounding boxes (BB) task that uses the camera (x,y) coordinate of the BB between the previous and current frames. It is based on the observation that there are small positional changes for moving objects within images in adjacent frames taken at a high frame rate. At 10 frames/sec objects will maintain their relative positions in successive frames hence the bounding box of object A in previous frame will have camera location coordinates closest to the bounding box of object A in the current frame. As only a comparison is required the L1 norm is calculated between bounding boxes. Code is implemented at `line # 237 in camFusion_Student.cpp`

```
    int currID = -1;
    for (std::vector<BoundingBox>::iterator it1 = prevFrame.boundingBoxes.begin(); 
          it1 != prevFrame.boundingBoxes.end(); ++it1)
    {
        double minDist = 1e8;
        int prevID = (*it1).boxID;
                
        for (std::vector<BoundingBox>:: iterator it2 = currFrame.boundingBoxes.begin(); 
              it2 != currFrame.boundingBoxes.end(); ++it2)
        {
            //Check L1 norm distance between bounding boxes within adjacent frames for a match 
            double minBoxDist = abs((*it1).roi.x - (*it2).roi.x) + abs((*it1).roi.y - (*it2).roi.y);
            if (minBoxDist < minDist)
            {
                minDist = minBoxDist;
                currID = (*it2).boxID;
            }
        }
        bbBestMatches.insert({prevID, currID}); 
    }
```    


### 2. Compute Lidar-based TTC
The Lidar based time-to-collision estimates were obtained by using the previous and current lidar points from the car directly infront of the ego car. Based on the Constant Velocity Model the TTC is given by d1 * dt / (d0 - d1) where:

d0 - distance measured from previous frame  
d1 - distance measured from current frame  
dt - elapsed time between frames  

To calculate the distance to the car a median value was used for each frame. This value reduced the error in  due to outliers. A helper function ,getMedian, was used to acheive this task. Code is implemented at `line # 200:216 and 220:234 in camFusion_Stdent.cpp `

```
double getMedian(vector<LidarPoint> lidarPoints)
{
    //Helper function to find the median distance, in the x-direction, of lidar points from 
    // objects directly infront of ego car.
    
    vector<double> LPoints;
    for (vector<LidarPoint>::iterator it1=lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        LPoints.push_back(it1->x);
    }

    std::sort(LPoints.begin(), LPoints.end());
    long medIndex = floor(LPoints.size() / 2.0);
    double medValue = LPoints.size() % 2 == 0 ? (LPoints[medIndex - 1] + LPoints[medIndex]) / 2.0 : LPoints[medIndex]; // compute median dist. ratio to remove outlier influence
    
    return medValue;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1/frameRate; // time between two measurements in seconds

    double medianXPrev =  getMedian(lidarPointsPrev);
    double medianXCurr = getMedian(lidarPointsCurr);

    // compute TTC from both measurements
    TTC = medianXCurr * dT / (medianXPrev-medianXCurr);
    
}
```

### 3. Associate Keypoint Correspondences with Bounding Boxes
The association of keypoint matches to the current bounding box is done at `line # 133 in camFusion_Student.cpp`


```
{
    for (vector<cv::DMatch>::iterator it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        //Check if the current bounding box contains the keypoint in kptsCurr indexed by kptmatches
        if (boundingBox.roi.contains(kptsCurr[it1->trainIdx].pt))
        {
            boundingBox.kptMatches.push_back(*it1);
        }
    }
    
}
```
The requirement to have all outliers removed based on the euclidean distance between keypoints within the bounding box is acheived in the ComputeTTCCamera function which is subsequently called.


### 4. Compute Camera based TTC
The outliers found in the current an previous keypoint vectors are removed by computing a ratio of the euclidean distance between adjacent current and adjacent previous keypoints which is used as the distance measure for that keypoint to calculate the TTC. Code is implemented at `line # 149 of camFusion_Student.cpp`

```
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    
}
```
### 5. Performance evaluation 1
#### Example 1 - The TTC for Figure 1 (14.091s) is less than that of Figure 2 (16.6894s) although Figure 1 precedes Figure 2 in the frame sequence as the vehicle decelerates getting closer to ego car. This discrepancy is as a consequence of the  median distance in Figure 1 being much closer to the Xmin value, based on the distribution of lidar points, which is closer to the ego car.

#### Figure 1
![alt text][image5]  
#### Figure 2            
![alt text][image6]  

#### Example 2 - The TTC for Figure 3 (12.8086s) is as expected greater than that of Figure 4 (8.9598s) as the vehicle decelerates and gets closer to ego car. However, a difference of > 3s is unexpected but is as a result of the lidar distance measurements being spread out over a larger range of values hence the median distance is higher.  
#### Figure 3
![alt text][image7]  
#### Figure 4
![alt text][image8]
### 6. Performance evaluation 2

|Detector Type|Descriptor Type|TTC Camera|TTC Lidar|TTC diff         |
|-------------|---------------|----------|---------|-----------------|
|SHITOMASI    |BRISK          |13.9019   |12.5156  |1.3863           |
|SHITOMASI    |BRIEF          |14.0747   |12.5156  |1.5591           |
|HARRIS       |BRISK          |10.9082   |12.5156  |1.6074           |
|HARRIS       |BRIEF          |10.9082   |12.5156  |1.6074           |
|FAST         |BRISK          |12.3541   |12.5156  |0.1615|
|FAST         |BRIEF          |11.2952   |12.5156  |1.2204           |
|BRISK        |BRISK          |13.4086   |12.5156  |0.8930|
|BRISK        |BRIEF          |12.8448   |12.5156  |0.3292           |
|ORB          |BRISK          |15.0326   |12.5156  |2.517            |
|ORB          |BRIEF          |21.3418   |12.5156  |8.8262           |
|AKAZE        |BRISK          |11.9055   |12.5156  |0.6101|
|AKAZE        |BRIEF          |13.2804   |12.5156  |0.7648|
|SIFT         |BRISK          |11.6767   |12.5156  |0.8389|
|SIFT         |BRIEF          |12.0906   |12.5156  |0.4249|  

#### Examples of TTC Camera estimates compared to TTC Lidar estimates are shown below.
![alt text][image0]  ![alt text][image2]  
![alt text][image1]  ![alt text][image4]  

Despite the use of the median distance there are NaN and -inf values from the ComputeTTCCamera function. Some detectors HARRIS and ORB produce very large differences between camera and lidar TTC. (see results/results_ttc-estimates.csv) 

There are some descrepancies in the camera TTC values generated as very small distances are permitted between the inner and outer previous keypoints when calculating the distance ratios which gives rise to large median distances in the ComputeTTCCamera function resulting in higher time-to-collision values.

The detector - descriptor pair that produced TTC estimates with the lowest mean differences in time over all frames when compared to the TTC Lidar estimates is SIFT-BRISK.

![alt text][image3]

