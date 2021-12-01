import requests, json, re, csv, os, pickle, optparse
import pandas as pd
import requests, sys

from bs4          import BeautifulSoup
from datetime     import datetime
from googlesearch import search


foldir  = os.path.dirname(sys.argv[0])
if "/" in foldir: foldir += "/"
datadir = foldir + "Data/" 

os.system('mkdir -p ' + datadir)



if __name__ == '__main__':

    usage  = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-d'    , dest='download', help='download portfolio', default=None, action='store_true')
    parser.add_option('-u'    , dest='update'  , help='update portfolio'  , default=None, action='store_true')
    parser.add_option('-b'    , dest='both'    , help='run both'          , default=None, action='store_true')
    parser.add_option('-o'    , dest='only'    , help='Run only'          , default=None)

    (opt, args) = parser.parse_args()
    script_run = []
    down_name  = "downloadPortfolio.py"
    upda_name  = "updatePortfolio.py"
    if   opt.both     : script_run = [down_name, upda_name]
    elif opt.download : script_run = [down_name]
    elif opt.update   : script_run = [upda_name]
    else:
        print "no running option set, please run either:"
        print "-d to download portfolio \n-u to update portfolio info\n-b for both"
        exit()
    onlySymbs = []
    doOnly    = opt.only
    if opt.only:
        onlySymbs = opt.only.split('_')
    print script_run
    for script in script_run:
        print "Processing "+script
        execfile(foldir+script)



