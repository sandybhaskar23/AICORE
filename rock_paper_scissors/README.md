Using https://teachablemachine.withgoogle.com/train/image to train a model which recognises rock, paper, scissors and when nothing happens

1st attempt I tried to use different possible ways each of the actions would look like to the camera.  With over 500 images it was expected to provide reasonable accuracy. The prediction was for scissors was extremely poor, followed by paper.  Both categories results in fluctuating confidence in the classifaction of the actions.

2nd attempt I reduced the hand variation of the 2 low prediction classes.  This reduced variation provide significant confidence in the prediction of the hand actions. 

With the 2nd attempt I hope to write an alogirithmn which I can play rock,paper & scissors against the computer. 

todo:
Build class which reads input? is it an image? 
Use a weight against each class 
