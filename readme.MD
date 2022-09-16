# Image based popular food items classification using AI and corresponding Web application deployment for end-users using Heroku

![A variety of foods](https://github.com/SohamBera16/foodetecto-An-AI-based-Food-category-detection-app-using-Streamlit-and-Heroku-Cloud/blob/main/istockphoto-922783734-612x612.jpg)

 * ### Problem Statement : 
In today's world, each and every one of us click thousands of images in our daily lives, and one of the most popular categories of these images is *food items*. No matter whether we go to a aesthetically appealing new restaurant or make a delicious meal in our home, we almost always make sure that the memory finds it place in our social media feeds. This massive amount of data can be leveraged by cutting-edge Artificial Intelligence technology to gain insights about the personal as well as regional/global trends of food consumption habits as well as various correlations with other factors. This knowledge can be extremely useful to figure out optimum diet plans, financial plans to tackle the overspending habits on junk foods, and how different food items contribute to the overall obesity factors to maintain our peak biological performance level.

Hence, the first and most important step of this process is to make the machines smarter in terms of visual perception and understanding/categorization capabilities for different & popular food items. This project attempts to make an algorithm so that a computer can automatically detect a certain food type from an image and create the framework for further analysis and knowledge discovery.

 * ### Challenges: 
 
 During Deployment: 
 1) Resolved python, pip, and tensorflow version incompatibility by performing effective modifications in the requirements.txt file 
 
 2) Problems with slugsize of the App 
 Solutions tried: a) usage of .slugignore to delete redundant files during runtime of deployment, b) purging build caches to reduce the slug size , c) using Heroku buildpack for python
 
 - *Final working solution*:  As Heroku cloud comes with a 500MB limitation and the new TensorFlow package is 489.6MB. To avoid dependencies and storage issues, I explored and figured out the change in the requirements.txt file which helped to reduce the slugsize in my use case - *Adding tensorflow-cpu instead of tensorflow reduced the slug size from 667.7MB to 377MB.*
 

 ![deployment](https://github.com/SohamBera16/foodetecto-An-AI-based-Food-category-detection-app-using-Streamlit-and-Heroku-Cloud/blob/main/app%20deployed.png)
 
 * ### Technology Implementation: 
Artificial Intelligence (AI)/Deep Learning techniques - in particular, the Inception V3 model has been used for implementing transfer learning along with the    Convolutional Neural network (CNN) tech in order to automate the process of food item image detection. Afterwards, the Heroku platform was utilized to deploy the app into production for the demo of automated image classification results. 
    
 * ### Dataset used: 
 The dataset used for this project has been downloaded from - https://www.kaggle.com/datasets/msarmi9/food101tiny
 Some of the example food items to classify are - apple pie, ice cream, falafel, etc. 
 
 * ### Demonstration of the app:
 
 ![web app](https://github.com/SohamBera16/foodetecto-An-AI-based-Food-category-detection-app-using-Streamlit-and-Heroku-Cloud/blob/main/streamlit%20web%20app%20deployment%20screenshot.png)
    
 * ### Results: 
 Accuracy of detection achieved on the training dataset is 84.25%. 
 The validation accuracy achieved on the test dataset was 86.13% which is quite good given that the analysis was performed on a comparatively very small dataset. 

### Future Steps:
