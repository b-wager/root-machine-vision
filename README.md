# Root Roost Detection with Haar Cascades
#### Computational Robotics Machine Vision Project
Brooke Wager and Kelsey McClung

## Intro
The goal of this project was to train a Haar Cascade Classifier to identify pieces of the board game, Root. In the end, we have trained a Haar Cascade Classifier to identify the roost pieces in photos of the game board. 

## How to run
### Test image on pre-generated cascade
To run an image on one of our pre-generated cascade files, run `test_haar_cascade.py` in the repo’s root directory. To change which cascade classifier is running, use the optional `--cascade` argument followed by the path to the cascade.xml file. To change the test image to use the model on, use the optional `--input` argument followed by the path to the image. You can also run the model on a live camera stream by using `--input 0` instead.

We have three different premade cascade files for detecting Root roosts.
1. Our first model was trained by annotating 15 positive images and comparing them to 56 negative images. This model did not perform very well and rarely ever made a guess. This meant that almost every roost was missed, so we did not gain any useful data from the model. However, we did notice that if the roost was held at a very specific angle, the model could detect it. This led us to train the next model.
2. Our second model was trained by an image of a single roost being rotated and transformed over our background images. We thought this model would perform much better due to the much larger volume of positive images and the variety of angles and perspectives of the roosts. However, this model ended up overguessing and tending to detect roosts at spots they were not at, while ignoring the actual roosts.
3. Our final model combined aspects of the first two models. We kept all of the annotated images, however, we also copied each image and transformed them at intervals of 3 degrees. This gave us a positive sample size that was 120 times larger than the first model while keeping the variability of different lighting and placements that were not included in the second model. This model ended up performing very well and was able to detect almost all roosts while only providing about 0-2 false positives per test image.

### Generate your own cascade file
#### Install necessary applications
To generate your own `cascade.xml` file, you will first need to install the OpenCV’s old Haar Cascade applications. You can easily do this with the following terminal commands:
```
git clone https://github.com/opencv/opencv
cd opencv
git checkout origin/3.4
git switch -c 3.4
sudo apt-get install cmake
mkdir build
cd build
rm ../CMakeCache.txt
cmake ..
Make
ls bin/*annotation*
ls bin/*create*
ls bin/*cascade*
```
If you would like to move the binaries to your system folder, so that you can later run `opencv_annotation` rather than
```
~/path/to/opencv/build/bin/annotation
```
you can run:
```
sudo mv /bin/opencv_annotation /usr/local/bin/
sudo mv /bin/opencv_createsamples /usr/local/bin/
sudo mv /bin/opencv_traincascade /usr/local/bin/
```
You should now have the tools needed to train Haar cascade models!

#### Steps to train Haar cascade models
1. Collect positive and negative images
- Negative images should not have any of your desired targets in the image.
- The images we used for roost-detection models are located in `~/cascade-training/images/roost-positive` and `~/cascade-training/images/roost-negative`.
- If you are planning to artificially generate images with `opencv_createsamples`, you only need one positive image. Our single roost image can be found in `~/cascade-training/images/test-images/single-roost.jpg`.
2. Create negative description file
- This is a .txt file that contains a list of all the background image files.
- Our negative description file for our roost cascade model is found at `~/cascade-training/roost_background.txt`.
3. Annotate positive images using `opencv_annotation` **(skip to step 5 if artificially generating images instead)**
- To create the .dat file used in `cascade-training/roost-train-cascade`, we used the command `opencv_annotation --annotations=roost_info.dat --images=images/roost-positive --maxWindowHeight=1000 --resizeFactor=2`.
- The `opencv_annotation` tool is very intuitive to use, and you can select multiple objects in each image.
4. *Optional:* Create transformed copies for each positive image
- If you would like your model to be able to detect objects more accurately at different angles, we recommend doing this step.
- First, open `cascade-training/train_images.py` and change the `input_file`, `output_file`, and `output_dir` to the desired paths.
- Then, in the `cascade-training` directory, run `train_images.py`.
5. Create vec file
- *For annotated images*: `opencv_create samples -info roost_info.dat -vec roost_vec` (where `roost_info.dat` is the file created with `opencv_annotation`).
- *For artificially generated images*: `opencv_createsamples -vec roost_vec_2 -img images/single-roost.jpg -num 1000 -w 24 -h 24` (where `images/single-roost.jpg` is the example object file.
6. Prepare the cascade directory
- In the `cascade-training` folder, `$ mkdir roost-train-cascade`.
- Copy our `params.xml` files from one of our train-cascade directories into your new directory.
7. Train the cascade model
`opencv_traincascade -data roost-train-cascade -vec roost_vec -bg roost_bg.txt -numNeg 56 -numStages 20 -w 24 -h 24 -featureType HAAR -numPos 1000`
- Change the -data, -vec, and -bg parameters to the desired files and directories.
- Change -numNeg to the number of negative images listed in the -bg file.
- Note: while training, the program will likely terminate due to an error. The last runs will have been saved, so you don’t have to restart. Just lower the `-numPos` value until the next stage starts running.

And that’s it! You now have a custom cascade.xml file that you can use to detect objects on images.

## What is Root?
Root is a board game with a default set of 4 players, each with different game pieces, as each player follows a unique set of rules to earn points and win the game. We became interested in using machine vision to identify different game pieces because of the variety of pieces, and the visually noisy game board that we expected would provide a fun challenge. We are also big fans of this game, so that helped convince us to pursue this project. The image below shows how the board might look mid game.

## Project Goals
The initial goal for this project was to create a program or train a model to identify and classify all Eyrie (blue bird) pieces, meaning both roosts and warriors. Roosts are flat square pieces, blue with a leaf logo in white. Warriors are 3D blocks, shaped vaguely like birds. 

Starting with this goal, we also had the idea of expanding our model/program to be able to detect other game pieces, with the lofty end goal of detecting all types of pieces and the areas of the board. This would allow it to know the game state from a photo. We are considering continuing with this project in the hopes of reaching this end goal.

## Image Classification
We looked at a few options for image classification/object detection. Initially, we were considering color detection, as game pieces are color coded to each player (Eyrie are blue, Cats are orange, etc). We also considered contour detection for recognizing the distinct shapes of certain pieces. Ultimately we choose to train a Haar cascade classifier. We believed color detection would not work on all game pieces, as some shared similar colors with the noisy background, and different types of pieces share colors as well. However, we did choose to start with Eyrie pieces because we felt the bright blue color had the most contrast with the board’s colors (we later learned that Haar cascade classifiers don’t use color). We didn’t choose contour detection because we felt the three dimensional block pieces would have different outlines from different angles, and ultimately we didn’t explore this path enough to understand how we could utilize it. We were very excited about the idea of training a cascade model, so we chose that method in order to learn more. We are happy with our choice for what we’ve learned, but one of those learnings is that Haar cascade is not the best method for identifying Root game pieces (at least when used alone).

## Haar Cascade Classifier
Haar Cascade was originally developed for facial recognition. It works by extracting “features” from positive training images (images that contain the object to be classified), then testing those features on their ability to classify the images.

The black and white boxes below visualize the features. Numerically, each feature is defined as the sum of pixels in the black box, minus the sum of pixels in the white box. These values can be used to understand the starkness of edges, lines, and four-rectangle areas.

Paul Viola and Michael J. Jones’ paper, *Robust Real-Time Face Detection*, shows this example of an edge feature used in face detection. In most face images, there will be a stark edge between the area of the eyes (black rectangle) and the tops of the cheeks (white rectangle). One can imagine how this feature between eyes and cheeks would be useful in identifying faces.

Back to training. The features with the lowest error rate in classifying the training images are selected to be used in the cascade model. Features are then organized into different stages which, together, make up the cascade model. 

To classify an image, the model runs one stage at a time over windows, or regions, of the image. Each region goes through the stages one by one. If any stage is failed, that region is skipped over and the stages start again on the next region. This saves lots of computation time, as some regions are quickly ruled out, and the most specific features are only run on regions likely to be the target object. If a region does pass all stages, it’s classified as the target object

## The Plan
Once we had landed on Haar cascade classifying as our method, we had a clear path forward:

1. Take positive and negative images of Eyrie and Roost pieces
2. Annotate images
3. Train the model
4. Test, and hopefully find success!

Note: We first completed this plan only using roost pieces, not warriors. In the end our warrior training was unsuccessful, so we will walk you through the roost training.

## Training Photos and Annotation 

We trained our first model on 15 positive images and 56 negative images. Positive images could contain one or more roost pieces. We annotated the positive images using OpenCVs annotation tool, denoting where in the image the roost piece(s) were. Having only used 15 positive images, we were very open to the idea that our model would not be very accurate, and that we would need to buff up our training data set.

The GIF below shows the annotations done on one of our positive training images.

## Training the model
Once images were annotated, we set up our files and used OpenCV’s createcascade and traincascade tools to train our first model. All models we trained for this project have 20 stages.

### Model 1
Our first model did not perform very well and rarely ever made positive classifications (whether accurate or not). This meant that almost every roost piece was missed, and that this model would not be reliable for identifying roost pieces. However, testing the model on live video, we noticed that if the roost was held at a very specific angle, the model could detect it. This led us to train the next model.

### Model 2
We believed our first model was struggling due to a lack of positive training images (15 is very little). So, we decided to use OpenCV’s image generation tool, `opencv_createsamples`. We used a real image of a single roost to generate a set of training images of the roost rotated and transformed over our background images. 

We expected this model to perform much better than Model 1 due to the larger volume of positive training images and the variety of angles and perspectives of the generated roosts. However, this model ended up overguessing, often detecting roosts where there were none, and missing the actual roosts. 

### Model 3 (Our favorite)
It was still clear to us that Model 1 suffered from too few training images, but generated images seemed unsuccessful for Model 2. 

For our final model, we combined aspects of the first two models. We reused all of the real, annotated images from Model 1. We also copied each image and transformed them at intervals of 3 degrees to create new training images. These transformations gave us a positive image sample size that was 120 times larger than the first model, and still only used real images, which are more successful at training than generated images. This model ended up performing very well and was able to detect almost all roosts while only providing about 0-2 false positives per test image.

## Side Quest: CRST Tracking
Between model training, we investigated CRST object tracking as a possible alternative. We quickly discovered that OpenCV’s CRST tracking tool would not suit our project needs, as the object to track had to be identified manually or by a separate tool before tracking began. We consider object tracking a future possibility for this project. Now that we have a classifier that identifies roost pieces, we could combine our cascade classifier with an object tracker to identify and track pieces through a video of gameplay. 

## Results analysis: Strengths and Weaknesses
Model 3 is a very hopeful success! When run on the training images, it has a hit rate of 1, meaning all roosts are detected in our training set. This does not mean it will never miss roosts in other images.

The false alarm rate on Model 3 is 0.786, meaning that 78% percent of identifications are **not** false alarms (false positives). This isn’t terrible, but false alarms will be a bigger issue for this model than false negatives. Many of the false alarms we observed in our testing photos were a very different size than the actual roosts in the photo, and were often found on parts of the board where roost pieces are never placed in real games. Moving forward, we believe we can reduce the amount of false alarms by implementing size and location thresholds. 

Our final model is not perfect, but we believe that it and our learnings from this project have set us up well for potential future work. Our final model only identifies one kind of Root piece out of many, and will benefit from additional thresholds and identification methods, such as size and color.

## Challenges
Our biggest challenge was setting up the requisites and code structure for this project. We followed a very helpful OpenCV tutorial [here](docs.opencv.org/4.12.0/dc/d88/tutorial_traincascade.html), but many of the tools we wanted to use were difficult to set up on our devices. For example, ‘pip install opencv’ does not install the functions for annotation, createsamples, or traincascade. We had to download these apps from OpenCV’s github and build them using cmake on our devices. We followed this useful tutorial to figure that out [here](https://github.com/bilardi/how-to-train-cascade)

## Lessons Learned
As mentioned earlier, we initially choose to start with identifying the roost (and warrior) pieces due to their bright blue color. After training images had been taken and annotated, we learned that Haar cascade classifiers don’t use color. This was a little embarrassing for us, but it provided a good lesson about assumptions.

We also learned a lot about different ways of creating training images,as we used a different method for each model, and got to see the benefits or drawbacks of each. In general, we gained a new understanding of how classifiers are trained and what makes good training data. 

Learning about Haar cascade classifiers specifically was very interesting as well. We began this project with little understanding of how Haar cascade works. After reading this [Geeks For Geeks article](https://www.geeksforgeeks.org/python/detect-an-object-with-opencv-python/), we understood the potential of the Haar cascade but not its limitations. As we’ve learned more about the workings of our model, we now have a better understanding of why our roost training was successful, but our warrior training was not. 



## Future Work
We started to try to implement this model for the 3D warriors pieces, but it did not work at all. We don’t think there is enough contrast between greyscale warrior pieces and the greyscale game board for the Haar Cascade model to be a reasonable tool for the detection of these objects, and we will likely switch to something color-based for this task. However, we would like to create Haar Cascade models for the other flat Root pieces, since it seems to be good at detecting the identical symbols on them.

Remember that Haar cascade uses features, which are calculated as the sum of pixels in one area minus the sum of the pixels in an adjacent area. This means that edges and lines of high contrast are strong features. The Roost pieces, a white logo on a flat dark blue square, are much better suited for Haar cascade than the warrior pieces, which are one solid color and can have a variety of silhouette shapes depending on the angle of the photo.

Another option is to filter by color. After we detect the game pieces with our model, we could filter out the false positives by comparing the colors of all the detected “pieces” to what we expect. We could also try out other recognition models that do take color into consideration.

Finally, we would like to be able to store a representation of the current game state on a computer from a picture. This would require knowing where each clearing is on the board and being able to detect the number and type of pieces in each one. If images are 
