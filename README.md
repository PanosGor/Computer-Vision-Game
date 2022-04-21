# Rock Paper Scissors Game
The goal of this project is to create a Rock, Scissor, Paper game in python, 
which will be processed in a live video and it will be able to give instant results to the users. 
The main idea is to create a program, which turns on the camera, creates two boxes which are the points of interest and then the two players put their hands inside, 
having Rock, Paper or Scissor displayed. 
Then the program will be able to recognize these gestures and it will print the result of the winner. 
This is not an easy task, because the computer cannot recognize human hands or specific signs that a human’s hand can create
Some difficulties that might occur are the variety of lighting conditions, the different backgrounds, the hand gesture identification and more

A lot of similar research can be found around this topic and there is a big variety of approaches and methods for the same problem


**Two different methods were used for the hand/sign detection.**

The two sub methods were used parallel to each other, the first for player 1 and the second for player 2.

*First Approache*
![image](https://user-images.githubusercontent.com/82097084/164471035-c3323f2a-2e44-4dcd-8554-5c09e9499b01.png)

The first step was to create a method that identifies a region where the user may put in her hand and also recognizes that a hand has entered this region.
To achieve that, two Regions of Interest were identified for each hand (ROI_1, ROI_2). 

Folder "Methods" contains the dataset and the two Python scripts that were ysed for gathering the data and training the model 

rock_papper_scissors.py is Main program to run for playing the game no changes in the code required for the program to run

rock_papper_scissors_model.h5 is the trained CNN model that needs to be included in rock_papper_scissors.py

in order for rock_papper_scissors.py to run properly both rock_papper_scissors_model.h5 and rock_papper_scissors.py need to be in the same directory 
