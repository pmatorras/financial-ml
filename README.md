# Stonks
Small framework that takes your portfolio from degiro, and gets the actual and expected values as well as the recommendations from either WSJ or cincodias (if spanish stock) public webpages.
In order to get the framework running, change the config_template to your username and password, and move it to config.


Stock plotter drawStonks.py, the following dependencies are required:
	     pip install plotly
	     pip install yfinance
	     pip install -U Kaleido
	     
Two different plotting options are currently available:

-f: Show forecasts, two plot types are done:
  
  a) Current price and the max/min for the last 52 weeks is shown together with the expectated 12 month value and the BEP
  
  b) Diagrams "expert" buying/selling recommendations for each of the stocks are shown. 

-s: Show current price for each product (-s).

Both cases can be run at once with the -b option

Other avaliable options include:

-i: Draw plots interactively

-t: Run on test option

-o: If only some plots from the portfolio are desired

-p: Specify the period over which the plot is done (ytd if default)