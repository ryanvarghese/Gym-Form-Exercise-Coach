# Gym-Form-Excercise-Coach


## Table of Contents
* [About the project](#about-the-project)
* [Installation](#installation)
* [Diagrams](#diagrams)
* [Result and Analysis](#result-and-analysis)
* [Technology Used](#technology-used)
* [Authors](#authors)
* [Credits](#credits)
* [Refrences](#refrences)


## About the project
- The ongoing pandemic has made all of us realize the significance of a healthy lifestyle. This has encouraged many people to join gyms, but however, it is also true that executing an improper form of exercise may lead to injuries. A common observation is that even people who visit the gym regularly find it difficult to perform all steps (body pose alignments) in a workout accurately. Continuously doing an exercise incorrectly may eventually cause severe long-term injuries. 

- To help solve this problem and provide assistance in form of visual feedback while performing a workout. We represent the human body as a collection of limbs and analyse angles between limb pairs to detect errors and provide corrective action to the user.

- Our model provides a skeletal grid of the person performing the exercise v/s ideal-case scenario. This will help the user to self analyse his/her precision for a particular exercise. Also, it will provide the user with ‘specific feedback’. Specific feedback refers to describing what the individual did correctly rather than providing a general statement such as saying, “good job.”

- **We have used the MoveNet Model to detect key points. We then draw the key points and edges on the user’s body. Now we calculate the limb angles and compare them with our ideal case scenario angles and then we display the feedback to the user. We train the YOLO algorithm on our dataset. Then we run our model on the user’s video to detect and trace the path of the barbell. We draw the boundaries of the path which if the user crosses the boundaries will change its color**. Depending on which boundary the user crosses, accordingly, the appropriate feedback would be given to the user. 

- The output video is processed and sent back to the user using an API developed in FAST API. This video is displayed on the HTML page developed for the same. 


## Installation
### Dependencies 

## Diagrams
### **Architecture**
![Architecture](https://user-images.githubusercontent.com/22417910/197022224-6a20b8cb-b83a-4eb0-befa-7d310824a451.png)

### **Activity**
![Activity Diagram](https://user-images.githubusercontent.com/22417910/197022415-f9006fc1-ee2f-4de1-8783-5b2033dccc23.png)

### **DFD**
![DFD Level 0 ](https://user-images.githubusercontent.com/22417910/197022441-20c11ed7-b4d6-428d-a389-adb53add6c32.png)

![DFD Level 1](https://user-images.githubusercontent.com/22417910/197022747-962332f2-1788-4927-a063-29067c9dca9e.png)

![DFD Level 2](https://user-images.githubusercontent.com/22417910/197022740-7c8f9ed7-45e6-4c7e-85df-53e8b78ca32f.png)



## Result and Analysis

## Technology Used

## Authors


## Credits
- [YOLOV5 Notebook](https://colab.research.google.com/drive/1gDZ2xcTOgR39tGGs-EZ6i3RTs16wmzZQ)
- [Movenet Model](https://www.tensorflow.org/hub/tutorials/movenet)

## References
- [Paper 1](https://www.researchgate.net/publication/324759769_Pose_Trainer_Correcting_Exercise_Posture_using_Pose_Estimation)
- [Paper 2](https://ieeexplore.ieee.org/abstract/document/8856547?casa_token=s7vlnKtOVoMAAAAA:kGlGEIAyXNAD7pmu6zuPOQdFNiLkenuaj3D_z-JKDixd0Nxnmi5iiS1e_cdkRkQ0hHaPHX55-JDuTF0)
- [Paper 3](https://arxiv.org/abs/1907.05272)
