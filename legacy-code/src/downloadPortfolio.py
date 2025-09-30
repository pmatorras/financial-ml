from requests.exceptions import HTTPError
if __name__ == '__main__':
    
    #Retrieve username and pasword from config file
    config_path = os.path.join(srcdir, "..", "cfg", "config")
    with open(config_path) as configFile:
        CONFIG = json.load(configFile)


    # login
    URL_LOGIN = "https://trader.degiro.nl/login/secure/login"
    payload   = {'username': CONFIG['username'],
                 'password': CONFIG['password'],
                 'isPassCodeReset': False,
                 'isRedirectToMobile': False}
    header    = {
        'content-type': 'application/json',
        'User-agent': 'Mozilla/5.0'}
    rlogin    = requests.post(URL_LOGIN, headers=header, data=json.dumps(payload))
    print(rlogin)
    sessionID = rlogin.json()["sessionId"]
    print(sessionID)
    # get int account
    URL_CLIENT = 'https://trader.degiro.nl/pa/secure/client'
    payload    = {'sessionId': sessionID}
    proxies = {
        "http": None,
        "https": None,
    }
    try:
        session = requests.Session()
        session.trust_env= False
        rclient    = session.get(URL_CLIENT, params=payload, proxies=proxies, headers=header)
        rclient.raise_for_status() 
    except HTTPError as e:
        print('error ocurred', e)
    intAccount = rclient.json()["data"]["intAccount"]

    #Retrieve data
    URL = "https://trader.degiro.nl/trading/secure/v5/update/"+str(intAccount)+";jsessionid="+sessionID
    payload = {'intAccount': intAccount,
               'sessionId': sessionID,
               'cashFunds': 0,
               'orders': 0,
               'portfolio': 0,
               'totalPortfolio': 0,
               'historicalOrders': 0,
               'transactions': 0,
               'alerts': 0}
    rcash = requests.get(URL, params=payload, headers=header)
    data  = rcash.json()

    # get cashfund
    cashfund = {}
    for currency in data["cashFunds"]["value"]:
        for parameter in currency["value"]:
            if parameter["name"] == "currencyCode":
                code = parameter["value"]
            if parameter["name"] == "value":
                value = parameter["value"]
        cashfund[code] = value


    ## get portfolio
    temp_portfolio = []
    for position in data["portfolio"]["value"]:
        to_append = {}
        for position_data in position["value"]:
            if "value" in position_data:
                to_append[position_data["name"]] = position_data["value"]
        temp_portfolio.append(to_append)

    portfolio = list(filter(lambda x: x["positionType"] == "PRODUCT" and x["size"]>0 , temp_portfolio))


    BEP = {}
    for fund in portfolio:
        BEP[fund["id"]] = { "BEP"       : fund["breakEvenPrice"],
                            "size"      : fund['size'],
                            'price'     : fund['price'],
                            'gain/loss' : round(abs(fund['todayPlBase']['EUR'])-abs(fund['plBase']['EUR']),4)}
    #product info
    url        = "https://trader.degiro.nl/product_search/secure/v5/products/info"
    payload    = {'intAccount': intAccount, 'sessionId': sessionID}
    pid        = [x["id"] for x in portfolio]
    r          = requests.post(url, headers=header, params=payload, data=json.dumps(pid))
    extra_info = r.json()

    #Add extra info and save
    for add_i in extra_info["data"]:
        for key in BEP[add_i]:
            extra_info["data"][add_i][key] = BEP[add_i][key]

    with open(full_port, 'w') as f:
        json.dump(extra_info["data"], f, indent=4)

