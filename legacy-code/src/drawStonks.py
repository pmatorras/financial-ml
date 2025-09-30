import matplotlib.pyplot as plt
from variables  import *
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

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
    else:
        print "no running option set, please run either:"
        print "-s for stock charts\n-f for stock forecast\n-b for both"
        exit()
    onlySymbs = []
    doOnly    = opt.only
    if opt.only:
        onlySymbs = opt.only.split('_')
    if opt.test:
        onlySymbs  = ["VOW3", "PRX"]
        doOnly     = True
    print script_run
    for script in script_run:
        print "Processing "+script
        execfile(srcdir+script)
