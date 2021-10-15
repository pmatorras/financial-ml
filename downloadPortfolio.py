import requests, json

#Retrieve username and pasword from config file
with open('config', 'r') as configFile:
    CONFIG = json.load(configFile)


# login
URL_LOGIN = "https://trader.degiro.nl/login/secure/login"
payload = {'username': CONFIG['username'],
           'password': CONFIG['password'],
           'isPassCodeReset': False,
           'isRedirectToMobile': False}
header = {'content-type': 'application/json'}
r = requests.post(URL_LOGIN, headers=header, data=json.dumps(payload))

sessionID = r.json()["sessionId"]

# get int account
URL_CLIENT = 'https://trader.degiro.nl/pa/secure/client'
payload = {'sessionId': sessionID}
r = requests.get(URL_CLIENT, params=payload)
intAccount = r.json()["data"]["intAccount"]

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
r = requests.get(URL, params=payload)
data = r.json()

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
            to_append[position_data["name"
            ]] = position_data["value"]
    temp_portfolio.append(to_append)
portfolio = list(filter(lambda x: x["positionType"] == "PRODUCT" and x["size"]>0 , temp_portfolio))
BEP = {}
for fund in portfolio:
    print fund["id"],fund["breakEvenPrice"]
    BEP[fund["id"]] = fund["breakEvenPrice"]

print BEP

## get product info
url = "https://trader.degiro.nl/product_search/secure/v5/products/info"
payload = {'intAccount': intAccount, 'sessionId': sessionID}
pid = [x["id"] for x in portfolio]
r = requests.post(url, headers=header, params=payload, data=json.dumps(pid))
additional_info = r.json()

print "rdict", r

json_object = json.dumps(BEP, indent = 4)  
#BEP_JSON = json.dump(BEP, indent=4)
#print "portfolio", portfolio

for info in additional_info["data"]:
    this_info =1# additional_info["data"][info].update(json_object[info])
#additional_info.update(portfolio)
with open('full_portfolio.json', 'w') as f:
    json.dump(additional_info["data"], f, indent=4)
    

#print dict.fromkeys(
for i in range(0,len(pid)):
    this_stock = additional_info["data"][pid[i]]
    print this_stock["name"], this_stock#
    ["symbol"]
