The source code in this directory was done in Python 3.6
Analysis was done using the latest version of scikit-learn, a Python machine learning library. For attribution purposes, articles required by scikit-learn are cited:

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013.

Installation:
To run the code in this project, make sure you have matplotlib, numpy, sklearn, and scipy installed

Top level folder overview:
datasets/ - contains the two datasets used for this project; in csv form
analyze_aviation_accidents.py - contains all the code needed to run analyses/experiments on Aviation_Accident_Database_&_Synopses.csv - could generate up to 8 csv files
analyze_coding_survey.py - contains all the code needed to run analyses/experiments on 2016_New_Coder_Survey.csv - could generate up to 8 csv files
wma61-analysis.pdf - contains the written report derived from the performed analyses/experiments

How to use:
- The code contains commented headers which describe what each section of comments does, be it KNN analysis looking for k, or AdaBoost pruning
- The code contained under a commented header and prior to another or the end of the file is modular and has no requirements of its own, only the initial file reading logic that precedes "# Analyze model complexity curve for NonBoost tree classifier, max_depth"
- Simply uncomment the section of code you'd like to run, corresponding to an analysis, and run the python file

How to run a Python file:
- on Windows: python "filename".py - with no quotations
- on Mac: python3 "filename".py - with no quotations

To uncomment the necessary code in the source files (it will look like below):
 '''code
	code
	code
	code'''
	
Simply remove the three apostrophes (or single quotes) on both ends to uncomment a section of code

