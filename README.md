# Deep-Fake-Detection

- Dataset:

  - The dataset can be found and downloaded from: https://www.kaggle.com/competitions/deepfake-detection-challenge/data
  - The code expects it to be put under a folder called /data

- Project rules:

  - The project accounts the 40 % of your grade (working code + written report + presentation).
  - You will get a grade based on the novelty, quality and complexity of the project, along with the ability to show use of appropriate models, preprocessing techniques and evaluation strategies.
  - The projects are done in groups of 2-4.

- Please see project guidelines.

- These are the list of projects suggested but you are free to choose your own topic as well:
  - Deep fake detection challenge
    - We will use the dataset from https://www.kaggle.com/competitions/deepfake-detection-challenge/overview to an external site.
    - Explore different deep neural network architectures
    - Copying existing notebooks will result in disqualification
    - The solution should be using tensorflow APIs or pytorch or any other framework

# TODO

- Decide the topic
- Specify the preferences here
- Deadline for registering the project is 20.03.2023 (11.59 pm).
- Register your team in Canvas (group registration)
- Check if know the task clearly,
- Is it supervised? or unsupervised task?
- Pattern detection or anomoly detection?
- Write down the task description, input, expected output, potential challenges you expect
  - if you have the appropriate data for the task
  - Look for multiple datasets preferably
  - if it is supervised task do you have labels?
  - If you want to use deep learning do you have enough data?
- Decide your roles in the project
- Divide tasks team work is important
- Roughly each member should contribute for 1/3rd of the load (or whatever your team size is)
- Cleanup, Explore the data
  - Remove noise if any
  - Check for outliers
  - Normalize, feature scaling, binarize see what is necessary
  - apply SVD or PCA or TSNE to visuzalize
- Decide which data mining algorithm is suitable?
  - Possibly several
  - Try a few algorithms you are free to use sklearn or whatever library you prefer
  - Possibly ensemble techniques like random forest or adaboost work better
  - Possibly a new method, or novel set of features for the same methods
- Experiments
  - paramter tuning
    - Learning rate, regurlarization parameter, dropout etc
    - Depth and width of neural network
    - Model specific parameters for CNNs and RNNS or any other type of neural network you are training
    - For unsupervised techniques cluster size etc
  - compare all the approaches
  - document the limitations of all the approaches you try
  - Choose the best performing method and do more detailed analysis
- evaluation measures
  - Precision, Recall, F1, ROC curve
  - Statistical significance test (see the examples here Links to an external site.)
- Report
  - At least 4 pages (max 10 pages) in 2 column ACM conference style in latex (Here is the link to the template on overleaf Links to an external site.)
  - Writing should be high quality technical writing not copy pasting think of this as a writing practice for your thesis
  - Describe the task, dataset, methods you use and discuss their limitations you saw in the experiments
  - Don't forget to include the github link to your code in the report along with a README.md in github repository which mentions how to reproduce your results in the report.
  - Very important: Document who did what and justify you did equal share of the tasks
- Presentation
  - 5-10 minutes presentation per team
  - Discuss the problem, data, interesting findings etc
  - In person project presentations are preferred.
- Deadline
  - 26.04.2023
