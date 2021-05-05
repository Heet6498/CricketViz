# CricketViz

### Introduction
Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. Here in this End-to-End Data visualization Project I scraped data from the ESPNCricinfo website. This site contains historical cricket data. With the help of various tools I programmed  this End-to-End Data visualization Project. 

### Tools
- SQLite
- Pandas(Python)
- Streamlit
- heroku

### Approch
#### - Dataset
The table contains fields such as Teams, Results, Margin, BR, Toss, Bat, Opposition team, Ground, and Year.
- Data scraping from ESPNCricinfo
- Data cleaning using SQLite and Python

#### - Data cleaning using SQLite and Python
The scrapped data have many jumbled rows. To make the data more understandable, I used the SQLite tool and DB browser as my SQLite tool.

#### - Producing visualizations using streamlit on a local server
Streamlit is an open-source Python library that helps to create and assign attractive, custom web-apps for machine learning and data science. In just a few moments you can build and deploy powerful web apps.
- libraries
- Defining the multi-select functionality
- Defining the dropdown menu
- Defining the side-bar
- Count plots
- Custom functions

#### - Deploying web-app on Heroku
- requirements.txt
- setup.sh
- Procfile

### Conclusion
Streamlit is a very useful and fast service when it comes to creating interactive web applications. In this article, we created one end-to-end data visualization web-app. We saw that how easy it is to define all filer features and how fast is it to deploy the web-app using Heroku.

### Future scope
We can create some machine learning models to predict the resulting cricket match outcomes given all these filters and add predicting results to this web-app.

Article: https://soniheet6498.medium.com/end-to-end-data-visualization-web-app-using-streamlit-from-data-scrapping-to-deployment-98beb963df57

Webapp: https://cricketviz.herokuapp.com/



