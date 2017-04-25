Any source code I have written was all done in Java 8 (and one Python 3.6 file) - if you'd like to see specifically which pieces of code I wrote, visit: https://github.com/willzma/ABAGAIL and view my commit history there
Otherwise, all cther code and analysis was done using the latest version of ABAGAIL;

Attributions:
Isbell, C. (2003). Randomized Local Search as Successive Estimation of Probability Densities (pp. 1-11, Tech.). Atlanta, GA.
ABAGAIL, an open-source Java machine learning library - https://github.com/pushkar/ABAGAIL

Installation (pulled from original ABAGAIL repo):
1. Install Java 8 SDK from here http://www.oracle.com/technetwork/java/javase/downloads/index.html
2. Install Ant http://ant.apache.org/
3. Clone or download source files from Git
4. Go with command line to where the build.xml file is and run: ant (note: the ant executable should be in your path somehow if you installed ant correctly.. so will java and javac)
5. Now run your scripts

How to use:
1. Enter the ABAGAIL folder
2. Open a Terminal, PowerShell, or other command line utility
3. Type "ant" (this will compile all the source files; make sure ant is installed!)
4. Select a test file to run from src/opt/test/
5. Run this command in your command line from the main directory: "java -cp ABAGAIL.jar opt.test.<TestName>" without quotes, and replacing <TestName> with your test of choice
6. Whatever test-specific code is in that file will execute

Where to find my files:
- Almost all of my code is prefixed with some kind of 2016_New_Coder_Survey or NewCoderSurvey or some variation of this
- All datasets are in the src/opt/test folder; all of them are csv files
- There is a Python file in the src/opt/test folder used to normalize my dataset with Pandas before use
- All my test files are in the src/opt/test folder; many of them are cloned versions of each other with one parameter changed; this was because I was running the experiments on multiple computers
- A number of ABIGAIL core files were changed to fit the needs of the analysis
- What certain test files do are easy to figure out if you remember what RHC, SA, GA, and MIMIC mean