from pytickersymbols import PyTickerSymbols
import logging
from typing import List, Dict
import json

class ticker_service:
    def __init__(self):
        self.stock_data = PyTickerSymbols()
        self.exchange_functions: Dict[str, str] = {
            'S&P 500': 'S&P 500',
            # 'FTSE 100': 'FTSE 100',
          
        }
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.base_tickers_store = None
        self.all_tickers()

    def get_tickers(self, index: str) -> List[Dict]:
        return self.stock_data.get_stocks_by_index(index)

    def get_all_tickers(self) -> Dict[str, List[Dict]]:
        all_tickers = {}
        for exchange, index in self.exchange_functions.items():
            try:
                tickers = list(self.get_tickers(index))
                all_tickers[exchange] = tickers
                self.logger.info(f"{exchange} tickers: {len(tickers)}")
            except Exception as e:
                self.logger.error(f"{exchange} generated an exception: {e}")
                all_tickers[exchange] = []
        return all_tickers

    def all_tickers(self):
        tickers = self.get_all_tickers()
        json_tickers = {}
        for exchange, ticker_list in tickers.items():
            json_tickers[exchange] = {ticker['symbol']: ticker['name'] for ticker in ticker_list}
        with open('all_tickers.json', 'w') as json_file:
            json.dump(json_tickers, json_file, indent=4)
        self.base_tickers_store = json_tickers
        self.logger.info("All tickers have been saved to 'all_tickers.json'")
        return json_tickers