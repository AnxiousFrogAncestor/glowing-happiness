import yfinance as yf
from rich import print
import pandas as pd

def get_prices(ticker:str, start_date:str, end_date:str, interval:str="1wk"):
    """Return adjusted price, which accounts for stock splits, dividends, etc.

    Args:
        ticker (str|list[str]): ticker symbol
        start_date (str): start date of stock price
        end_date (str): end date of stock price (e.g. 2025-01-01)
        interval (str): time-granularity of the data e.g. 1wk, 1d

    Returns:
        pd.DataFrame: The adjusted closing price for the given interval and between the given dates.
    """
    data = yf.download(ticker, start_date, end_date, interval=interval, auto_adjust=True)
    return data['Close']

def gather_links(search_term:str):
    """
    Fetches news article titles and links for a given company name using Yahoo Finance.

    Args:
        search_term (str): The name of the company to search for (e.g., "Microsoft").

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the 'title' and 'link'
        of a news article related to the company.
    """
    dat = yf.Search(search_term)
    links = []
    for news in dat.news:
        links.append({"title": news["title"], "link": news["link"]})
    print(links)
    return links

import pandas as pd
import urllib.request

def lookup_ticker(company_name:str):
    """Returns the best matching ticker symbol from the company name.

    Args:
        company_name (str): The name of the company.

    Returns:
        str: Best match security's ticker symbol

    #TODO add asserts
    """
    url = f'https://finance.yahoo.com/lookup/equity/?s={company_name}'

    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        html = response.read()

    tables = pd.read_html(html)
    security_symbol = tables[0]["Symbol"]
    return security_symbol


from finqual import finqual as fq

def get_fundamental_data(company_name:str, year_start:str, year_end:str, fundamental_data_type:str="income"):
    #get_fundamental_data("NVIDIA", year_start=2020, year_end=2025)
    #TODO add more fundamental data
    ticker_name = lookup_ticker(company_name)[0]
    if fundamental_data_type == "income":
        result = fq.Ticker(ticker_name).income(year_start, year_end)
        #print(type(result))
        return result
    elif fundamental_data_type == "balance":
        result_df = fq.Ticker(ticker_name).balance(year_start, year_end)
        #print(type(result), result)
        result = result_df.T.to_dict(orient="index")
        return result

#get_prices("AAPL", start_date="2020-01-01", end_date="2021-01-01")