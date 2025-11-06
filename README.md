All the results, dataset , analysis derived from this code are in this folder:
https://drive.google.com/drive/folders/1pyxxOi1uXEYAmjgmzWTYUklfEzcaANt6?usp=sharing

you can get all the results from this link.
some folders files are in zipped format. just unzipped them in the same folder.

Here is complete guide for you. how you can setup everything.
1. git clone this repository and place in your favourite directory.
2. rename folder from 'AGAROSE-HYDROGEL-TRENDS-USING-AI-MLl' to 'AGAROSE-HYDROGEL-TRENDS-USING-AI-ML' to avoid the path conflicts.
3. download all the analysis files from given google drive link and place those folders in the 'AGAROSE-HYDROGEL-TRENDS-USING-AI-ML' folder in same order to avoid the path conflicts.
4. In google drive 'CODE->DL MODELS (copy)' folder there are also necessary dataset and weights folder that also necessary to run the code.
5. navigate to the code folder there is separate folders of code for each analysis.
6. you can run 'train_all_model.py' file to train all the models.

The final directory should looks like this to avoid any error path related issues:

AGAROSE-HYDROGEL-TRENDS-USING-AI-ML-> [STED] Internal 0.375% 
                                  |-> Accuracy 
                                  |-> AFM Accuracy INTERNAL
                                  |->  ...
                                  |-> CODE -------------------------------->BASE_DIR
                                  |-> ...                                |->BUBBLE ANALYSIS CODE
                                                                         |->DL MODELS (copy)---------------------------------> Analysis Results 1
                                                                         |->FIBER ANALYSIS CODE                            |-> Analysis Results 2
                                                                         |-> ...                                           |-> Dataset
                                                                                                                           |-> Labels
                                                                                                                           |-> models
                                                                                                                           |-> weights
                                                                                                                           |-> ...
                                   

