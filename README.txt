#H1 Rock Paper Scissors Game
The goal of this project is to create a Rock, Scissor, Paper game in python, 
which will be processed in a live video and it will be able to give instant results to the users. 
The main idea is to create a program, which turns on the camera, creates two boxes which are the points of interest and then the two users put their hands inside, 
having Rock, Paper or Scissor displayed. 
Then the program will be able to recognize these gestures and it will print the result of the winner. 






Folder "Methods" contains the dataset and the two Python scripts that were ysed for gathering the data and training the model 

rock_papper_scissors.py is Main program to run for playing the game no changes in the code required for the program to run

rock_papper_scissors_model.h5 is the trained CNN model that needs to be included in rock_papper_scissors.py

in order for rock_papper_scissors.py to run properly both rock_papper_scissors_model.h5 and rock_papper_scissors.py need to be in the same directory 

