# Sentiment Analysis -- (Lexicon Approach)
This repository maintains the code for scoring raw text of comments/raw text using NLTK's VADAR algorithm, NLTK's VADAR base lexicon, and a custom healthcare lexicon.


**Setting up the virtual conda environment:**

1. Clone/download repository to your local machine 
2. cd (change directory) to location of cloned repository (ex. ```cd Users/.../sentiment_analysis``` )
3. Open Anaconda Prompt
4. Run the command ```conda create --name sentiment_analysis python=3.9```
5. Run the command ```activate sentiment_analysis``` in terminal 
6. cd (change directory) to location of cloned repository (ex. ```cd Users/.../sentiment_analysis``` )
7. Run the command ```pip install -r requirements.txt``` in terminal. This will install the necessary packages to run the scripts
8. Run the command ```deactivate``` in terminal. 


**Important Information about the Lexicon:**
- Additional healthcare lexicon keywords can be added to the health_lexicon.py file.


**Running the Notebook:**

1. Enter the virtual environment with the following command: ```activate sentiment_analysis```
2. Install NLTK files. This may require setting proxy : ```nltk.set_proxy(f'http://{USER}:{PASSWORD}@proxy:port')```
3. Open and run the Jupyter notebook. The sql query may take a long time to load. To test it out, adjust the sql query to only return xx number of records.

**USAGE**

**Input: 2 Arguments**

1. ```Dataframe (with comments column)```
2. ```Name of comments/text column (str)```

**Output: Dataframe with additional columns**

1. ```SENTENCE_COMP``` : Dictionary output {sentence : score} for each sentence in comment/text
2. ```SENTENCE_NUM``` : Number of sentences in comment/text
3. ```SUM_SCORES``` : Sum of the individual sentence -- composite score
4. ```OVERALL_SCORE``` : Normalizes column of sum_scores to be between -1 and 1. Values range from most negative (-1) to most postive (1)
5. ```OVERALL_SENTIMENT``` : Lables "Positive" and "Negative" comments based on the overall_score. Scores greater than/equal to ZERO are "Positive" 
6. ```SENTIMENT_IND``` : Indicator column 1 or 0 based on overall_sentiment. 1-positive, 0-Negative
7. ```TEXT_WORDS``` : List of text tokens filtered for stopwords
8. ```VIOL_DSC``` : Identifies statistical anomalies. "Out of Control" or "In Control"
9. ```VIOL_IND``` : Indicator for statistical anomalies. 1-Out of Control, 0-In Control

```sentiment_analyzer.get_sentiment_analysis(dataframe, "comments_column")```

**Citiations**
Bird, Steven, Edward Loper and Ewan Klein (2009).
Natural Language Processing with Python.  O'Reilly Media Inc.

For questions please reach out to **Adam Heisey** at ```admrussel@gmail.com```
