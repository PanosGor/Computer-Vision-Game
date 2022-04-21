# Rock Paper Scissors Game
The goal of this project is to create a Rock, Scissor, Paper game in python, 
which will be processed in a live video and it will be able to give instant results to the users. 
The main idea is to create a program, which turns on the camera, creates two boxes which are the points of interest and then the two players put their hands inside, 
having Rock, Paper or Scissor displayed. 
Then the program will be able to recognize these gestures and it will print the result of the winner. 
This is not an easy task, because the computer cannot recognize human hands or specific signs that a human’s hand can create
Some difficulties that might occur are the variety of lighting conditions, the different backgrounds, the hand gesture identification and more

A lot of similar research can be found around this topic and there is a big variety of approaches and methods for the same problem


## Two different methods were used for the hand/sign detection.

The two sub methods were used parallel to each other, the first for player 1 and the second for player 2.

### First Approach
![image](https://user-images.githubusercontent.com/82097084/164471035-c3323f2a-2e44-4dcd-8554-5c09e9499b01.png)

The first step was to create a method that identifies a region where the user may put in her hand and also recognizes that a hand has entered this region.
To achieve that, two Regions of Interest were identified for each hand (ROI_1, ROI_2).

*Method 1:*
The program updates a running average of the background values in the ROIs. More specifically, for differentiating between the background, it calculates the
accumulated weighted avg for the background and then subtract this from the frames that contain some object in front of the background that can be distinguished as foreground. 
This is done by calculating the accumulated weight for the first 60 frames. 
After the accumulated average for the background is calculated it is subtracted from every frame taken after the first 60 to find any object that covers the background. 
This will allow the program to detect new objects such as a hand entering the ROIs.
To do that first the program calculates the absolute difference between the background and the frame and then apply Binary Threshold in order to grab the contours from the image.

Once the hand enters the ROI, then the program can detect the change and apply some thresholding techniques to isolate the hand and the hand segment.
Binary thresholding is used to grab the hand segment from the ROI, in order to calculate the contour around the white hand against a black background.
Once we have a thresholded hand segment the Convex Hull method is used to draw a polygon around the hand.

![image](https://user-images.githubusercontent.com/82097084/164473462-f419f3c0-8ad2-4edf-a140-5d5d4ffd516d.png)

Then the center of the hand is calculated against the angle of the outer points of the polygon to infer a finger count.
Each point of the polygon ideally should count for a finger. By calculating the distance of the point from the center, the program identifies if the finger is extended.
Depending on whether the fingers are extended those points are closer or further away from the center of the hand.
In order to account for lines coming from the wrist (the points towards the bottom of the polygon) the most extreme points of the polygon are calculated (top, bottom, left, right).
Then the intersection of the extreme points is calculated as the center of the hand. 
Next the distance for the extreme point furthest away from the center of the hand is calculated. By using a ratio (0.8) of that distance as a radius, a circle is created around the circle.

Any points outside if the circle and far away from the bottom, count as extended fingers.

### Second Approach

For Player 2 a model was developed by using Convolutional Neural Networks.
The model is divided into three parts 
-	Creating the dataset, 
-	Training the model on the dataset 
-	and using the model in order to predict the data.

It would be better to develop the training and testing dataset instead of downloading one from the internet, in order to have more control over the model and understand the process better.

**Dataset Creation**

For the creation of the dataset *create_gesture_data.py* can be used in order to capture multiple hand screenshots for each hand sign and create a training and a testing dataset.
The first step was to create a ROI where the user would put in her hand in order to take the screenshots.
After that, the same methodologies from model 1 (described earlier) were used in order to calculate the background and grab the segment of the hand.
Once the hand is isolated the program starts saving the images in a local folder (The user can specify how many images/frames wants to capture).

![image](https://user-images.githubusercontent.com/82097084/164475824-6ae05fb9-d16a-48c8-8087-839867824491.png)

*Templates generated to be used for training and testing the CNN model*






Folder "Methods" contains the dataset and the two Python scripts that were ysed for gathering the data and training the model 

rock_papper_scissors.py is Main program to run for playing the game no changes in the code required for the program to run

rock_papper_scissors_model.h5 is the trained CNN model that needs to be included in rock_papper_scissors.py

in order for rock_papper_scissors.py to run properly both rock_papper_scissors_model.h5 and rock_papper_scissors.py need to be in the same directory 
