import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os, sys, json, optparse
today    = str(datetime.date(datetime.now()))#-timedelta(days=1)) 
foldir   = os.path.dirname(sys.argv[0])
if "/"  in foldir: foldir += "/" 
plotdir  = foldir+"Plots/"
foredir  = plotdir+"Forecast/"
os.system("mkdir -p " + foredir)



if __name__ == '__main__':


    usage  = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-f'    , dest='fcast'   , help='run forecast'     , default=None, action='store_true')
    parser.add_option('-s'    , dest='stock'   , help='run stock price'  , default=None, action='store_true')
    parser.add_option('-b'    , dest='both'    , help='run both'         , default=None, action='store_true')
    parser.add_option('-i'    , dest='interact', help='interactive plots', default=None, action='store_true')
    parser.add_option('-t'    , dest='test'    , help='test keys'        , default=None, action='store_true')
    parser.add_option('-o'    , dest='only'    , help='Run only'         , default=None)
    parser.add_option('-p'    , dest='period'  , help='Check plot period', default=None)

    (opt, args) = parser.parse_args()
    script_run = []
    currname = "plotCurrentPrice.py"
    forename = "plotForecasts.py"
    if      opt.both : script_run = [currname, forename]
    elif   opt.fcast : script_run = [forename]
    elif   opt.stock : script_run = [currname]

    onlySymbs = []
    doOnly    = opt.only
    if opt.only:
        onlySymbs = opt.only.split('_')
    if opt.test:
        onlySymbs  = ["VOW3", "PRX"]
        doOnly     = True
        print "only running on", onlySymbs

    for script in script_run:
        print "Processing "+script
        execfile(foldir+script)
