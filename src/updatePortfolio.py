
def makeSoup(link):
    print link
    request   = requests.get(link, headers=headers)
    return  BeautifulSoup(request.text,"lxml")



def loopqueries(query,recom,links):
    for j in search(query, tld='com', num =1, stop=1, pause =1):
        if recom.lstrip() not in j: continue 
        links.append(j)

def getLinksGoogle(site, stock, recom):
    links  = []
    symb_i = stock["symbol"]
    name_i = stock["name"  ].replace("SA","")
    query  = site+ symb_i+" "+name_i+" "+recom
    print "Stock", symb_i, "\t query", query

    loopqueries(query,recom,links)
    if len(links) == 0 :
        query = site + symb_i + recom 
        loopqueries(query,recom, links)
    if len(links) == 0 :
        query = site + name_i + recom 
        loopqueries(query,recom, links)
    if len(links) == 0 :
        key_w = None
        for word in name_i.split(' '):
            if symb_i in word.upper(): key_w = word
        if key_w:
            query = site+key_w+recom
            loopqueries(query,recom, links)
        print links, key_w
            
    print "links", links[0]
    return links


if __name__ == '__main__':
    stocks_csv = {}
    jsonDict   = open(full_port, "rb")
    portfolio  = json.load(jsonDict)
    jsonDict.close()

    for entry in portfolio:
        if "ETF" in portfolio[entry]["productType"]:
            print "ETF", portfolio[entry]["name"]
            continue

        stocks_csv[portfolio[entry]["symbol"]] =  [portfolio[entry]["name"], portfolio[entry]["isin"]]

        if "ES" in portfolio[entry]["isin"]:
            links = getLinksGoogle("site:cincodias.elpais.com/mercados/empresas/ ", portfolio[entry], " recomendaciones")
            stocks_csv[portfolio[entry]["symbol"]].append(links[0])
        else: 
            links = getLinksGoogle("site:https://www.wsj.com/market-data/quotes/ ", portfolio[entry], " research-ratings")
            stocks_csv[portfolio[entry]["symbol"]].append(links[0])




    f = open(port_pkl,"wb")
    pickle.dump(stocks_csv,f)
    f.close()
