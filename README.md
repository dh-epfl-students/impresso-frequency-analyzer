# Frequency analyzer
A project from [Christian Gasser](https://github.com/gasserchristian) in frame of the bachelor project (2019) of electrical engeneering at EPFL. Supervised by [Maud Ehrmann (@https://github.com/e-maud)](https://github.com/e-maud) and [Matteo Romanello (@mromanello)](https://github.com/mromanello) 
## Introduction
The goal of the project is to create an analyzer that it's build to see publication frequency variation of a set of journals in Europe.
The project is a dashboard that is build with a graph of publication interval (highlighted with specials points), a graph of the error mean (difference between model and reallity), a graph for the occurance of this intervals (to see the spectrum of how often an interval occurs) and a preview zone.

## Research summary
Initially, the goal of the project was to explore the metadatas of the journals. Finally, the idea was to create an analyzer of these metadatas that is global enough for explore many differents variables but we focus on the time interval between an issue and the next one in a journal.

The results are documented in the report. We can choose the journal we want to analyze in order to obtain the main view. It's somehow the signature of each journal. Then with the mean error we can explore the stability of the journal throught the years. We can also try to find some relationships between historical events and the changing of publication frequency. Finally, with a bit of experience it is possible to find outliers that show us a strike for example.

## Installation and Usage
The project is coded in Python. It use followings libraries :

* Dash, which needs :
 * plotly (graphing)
 * Flask (server)
* Pandas
* Numpy
* sqlalchemy

Once you install Dash, Pandas, Numpy and sqlalchemy and set the os system variables with yours access codes you have to run the server. You have to tip in the prompt after activating your environment this :
```
python server.py
```
The prompt gives you which ip and port you have to write in your Navigator. In our code, the IP is 28.178.115.26 and port 9102. But you have to see what ip your computer has and which port is free.

## License
This project is under the MIT licence.