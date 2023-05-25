# Prediction of anticancer drug combinations and their therapeutic effects by machine learning

The following instructions are for Linux operating system and Windows is currently not supported. All command lines start with a "$" symbol, which indicates Bash shell command-line prompt: these "$" symbols should be omitted when entering commands into your shell.

## I. INSTALLATION

The following data and software needs to be present in the your system before attempting to use drug combination predictor.

### 1. Download all files in this project to your folder.
    $`git clone https://github.com/jingbozhou/drug_combination_prediction.git`
    
### 2. Download required data
    Download Data.tar.gz (https://drive.google.com/file/d/1lDfHnsby79DQoabxNGfQbZFCc8SwTwJy/view?usp=share_link) and put them into CombDrugModule folder, uncompress it.
    $`mv Data.tar.gz <PATH_TO_Predictor>/drug_combination_prediction/`
    $`cd <PATH_TO_Predictor>/drug_combination_prediction/`
    $`tar zvxf Data.tar.gz`


### Anaconda3 (https://www.anaconda.com/download)

Brief training and predicting process:
1. 

2. Download Data.tar.gz (https://drive.google.com/file/d/1lDfHnsby79DQoabxNGfQbZFCc8SwTwJy/view?usp=share_link) and put them into CombDrugModule folder, uncompress it.

3. `python 1-get_train_model.py`: Get training model. It needs around 150G space to store training model.

4. `python 2-get_predict_result.py example_input.csv example_output.csv`: Get predict result.


The detail of installation package and process will publish soon.
