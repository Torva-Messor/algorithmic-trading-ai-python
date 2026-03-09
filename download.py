import json
import datetime
import yfinance as yf
import numpy as np

def main():
    download_ticker()
    download_news()
    download_macro()
    prepare_data()

## Download BTC-USD historical data from Yahoo Finance
## Minute resolution data for the last 60 days
def download_ticker():
    try:
        with open('BTC-USD_historical_data.json', 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}

    data = yf.download(tickers='BTC-USD', period='1mo', interval='5m')
    encoded = data.to_json()
    decoded = json.loads(encoded)
    
    ## Locate the Open price key dynamically to avoid yfinance MultiIndex issues
    open_key = next((k for k in decoded.keys() if 'Open' in k), None)
    close = decoded[open_key] if open_key else {}

    ## Merge with existing data
    for kj in existing_data:
        if kj not in close:
            close[kj] = existing_data[kj]
    
    ## Save to file
    with open('BTC-USD_historical_data.json', 'w') as f:
        json.dump(close, f, indent=4)

## Download Macro data (VIX, DXY, TNX)
def download_macro():
    macro_symbols = {
        'vix': '^VIX', 
        'dxy': 'DX-Y.NYB', 
        'tnx': '^TNX'
    }
    macro_data = {}
    
    for key, symbol in macro_symbols.items():
        data = yf.download(tickers=symbol, period='1mo', interval='5m')
        encoded = data.to_json()
        decoded = json.loads(encoded)
        
        ## Dynamically find the Open key
        open_key = next((k for k in decoded.keys() if 'Open' in k), None)
        macro_data[key] = decoded[open_key] if open_key else {}
            
    with open('BTC-USD_macro_data.json', 'w') as f:
        json.dump(macro_data, f, indent=4)

## Download BTC-USD news from Yahoo Finance
def download_news():
    ## Load existing news to avoid duplicates
    try:
        with open('BTC-USD_news.json', 'r') as f:
            news = json.load(f)
    except FileNotFoundError:
        news = []

    ## Download News for BTC-USD from Yahoo Finance
    new_news = yf.Ticker('BTC-USD').get_news(count=1000)
    if new_news is None: new_news = []
    
    ## Basic deduplication check
    existing_titles = {item['content']['title'] for item in news if 'content' in item}
    for item in new_news:
        if item['content']['title'] not in existing_titles:
            news.append(item)

    with open('BTC-USD_news.json', 'w') as f:
        json.dump(news, f, indent=4)

## Prepare data for training
def prepare_data():
    output = []

    ## Load data from files
    with open('BTC-USD_historical_data.json', 'r') as f:
        ticker = json.load(f)
    with open('BTC-USD_news.json', 'r') as f:
        news = json.load(f)
    with open('BTC-USD_macro_data.json', 'r') as f:
        macro = json.load(f)

    ## Helper to find the closest historical macro quote (handles weekends/overnights)
    def get_closest_past_value(macro_dict, target_ts_ms):
        valid_times = [int(k) for k, v in macro_dict.items() if int(k) <= target_ts_ms and v is not None]
        if not valid_times:
            return 0.0
        return macro_dict[str(max(valid_times))]

    ## Augment with Pricing and Macro data
    for item in news:
        if 'content' not in item:
            continue
            
        title   = item['content']['title']
        summary = item['content']['summary']
        pubDate = item['content']['pubDate']

        ## Convert pubDate to unix timestamp
        pubDate_ts = int(datetime.datetime.strptime(pubDate, '%Y-%m-%dT%H:%M:%SZ').timestamp())

        ## Round down to nearest 5 minutes
        index = pubDate_ts - (pubDate_ts % 300)  
        target_ms = index * 1000
        
        price = ticker.get(str(target_ms))
        future_price = ticker.get(str(target_ms + 300000))

        if price is None or future_price is None:
            print(f"Skipping entry with missing price data: title={title}, pubDate={pubDate}")
            continue    

        ## Extract Macro Values
        vix_val = get_closest_past_value(macro.get('vix', {}), target_ms)
        dxy_val = get_closest_past_value(macro.get('dxy', {}), target_ms)
        tnx_val = get_closest_past_value(macro.get('tnx', {}), target_ms)

        ## Calculate 1-hour Local Volatility (Standard Deviation of last 12 periods)
        prices_1h = []
        for i in range(12):
            historical_p = ticker.get(str(target_ms - (i * 300000)))
            if historical_p is not None:
                prices_1h.append(historical_p)
                
        local_volatility = float(np.std(prices_1h)) if len(prices_1h) > 1 else 0.0

        difference = price - future_price
        output.append({
            'title': title,
            'summary': summary,
            'pubDate': pubDate,
            'pubDate_ts': pubDate_ts,
            'index': index,
            'price' : price,
            'future_price': future_price,
            'difference': difference,
            'percentage': (difference / price) * 100,
            'global_vix': vix_val,
            'macro_dxy': dxy_val,
            'macro_tnx': tnx_val,
            'local_volatility': local_volatility
        })

    with open('BTC-USD_news_with_price.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__": 
    main()
