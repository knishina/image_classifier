# image_classifier

This project develops code for a to build a ML model to recognize different species of flowers.  The model utilizes a pre-trained network and was trained/validated using labeled data.  This project uses a GPU enabled workspace and was completed via the Udacity platform.  The following modules were used analyze/visualize and build a predictive model: `pil, pandas, numpy` for data munging, `matplotlib` for data visualization, and `torch, torchvision, scipy` for model building and its associated analyses.

### Data.
The data for this project is quite large - in fact, it is so large it cannot be uploaded to Github.  If you would like the data for this project, you will want download it from the workspace in the Udacity classroom.  Completing the project is likely not possible on your local unless you have a GPU.  The project involves 102 different types of flowers, where there ~20 images per flower to train on.  Finally, the  trained classifier will be used to predict the type for new images of the flowers.
![Different Flowers](https://github.com/knishina/image_classifier/blob/master/assets/Flowers.png)

<br />

### Output.
The input picture is presented, and the resulting predictions and corresponding probabilities are presented as a bar chart.
![Outcome](https://github.com/knishina/image_classifier/blob/master/Images_README/01.png)


