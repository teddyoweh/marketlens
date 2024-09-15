import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import ta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from config import config
from macros import ticker_service
import os
from openai import OpenAI
class StockAnalyzer:
    def __init__(self, ticker, company_name, db_path='stock_data.db'):
        self.ticker = ticker
        self.company_name = company_name
        self.config = config()
        self.stock_dict = ticker_service().all_tickers()
        # Combine S&P 500 and FTSE 100 stocks
        markets = ['S&P 500'] # 'FTSE 100'
        self.stock_dict = dict(list({ticker: name for market in markets for ticker, name in self.stock_dict[market].items()}.items())[:10])
        self.db_path = db_path

        # Initialize data containers
        self.stock_data = None
        self.company_info = None
        self.competitors = None
        self.financial_ratios = None
        self.financial_statements = None
        self.technical_indicators = None
        self.news_sentiment = None
        self.earnings_call_analysis = None
        self.competitor_data = None
        self.market_analysis = None
        self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)

        # Connect to the database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Create tables to store data
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_analysis (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                analysis_date TEXT,
                competitors TEXT,
                financial_ratios TEXT,
                financial_statements TEXT,
                technical_indicators TEXT,
                news_sentiment TEXT,
                earnings_call_analysis TEXT,
                competitor_comparison TEXT,
                market_analysis TEXT
            )
        ''')
        self.conn.commit()

    def data_exists(self):
        self.cursor.execute('SELECT 1 FROM stock_analysis WHERE ticker = ?', (self.ticker,))
        return self.cursor.fetchone() is not None

    def fetch_all_data(self):
        if self.data_exists():
            print(f"Data for {self.ticker} already exists in the database.")
            return

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.get_competitors): 'competitors',
                executor.submit(self.get_stock_data): 'stock_data',
                executor.submit(self.get_company_info): 'company_info',
                executor.submit(self.get_financial_statements): 'financial_statements',
                executor.submit(self.get_news): 'news'
            }
            for future in as_completed(futures):
                data_type = futures[future]
                setattr(self, data_type, future.result())

        # Process data that depends on the fetched data
        self.financial_ratios = self.calculate_financial_ratios()
        self.technical_indicators = self.calculate_technical_indicators()
        self.news_sentiment = self.analyze_news_sentiment(self.news)
        self.competitor_data = self.get_competitor_data()
        self.market_analysis = self.perform_market_analysis()

        # Fetch and analyze earnings call (if available)
        transcripts = self.get_earnings_call_transcripts()
        if transcripts:
            self.earnings_call_analysis = self.analyze_earnings_call(transcripts[0]['link'])

    def get_competitors(self):
        prompt = f"Given the following list of companies and their ticker symbols:\n\n"
        for ticker, name in self.stock_dict.items():
            prompt += f"{ticker}: {name}\n"
        prompt += f"\nIdentify the top 3 competitors for {self.company_name} ({self.ticker}) from this list. Return the result as a JSON array of ticker symbols."

        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant. Your task is to identify competitors for a given company. Provide your response in the following JSON format: [\"TICKER1\", \"TICKER2\", \"TICKER3\"]"},
                    {"role": "user", "content": prompt}
                ]
            )
            competitors = json.loads(completion.choices[0].message.content)
            return competitors
        except Exception as e:
            print(f"Error in determining competitors: {e}")
            return []

    def get_stock_data(self, period="1y"):
        stock = yf.Ticker(self.ticker)
        self.stock_data = stock.history(period=period)
        return self.stock_data

    def get_company_info(self):
        stock = yf.Ticker(self.ticker)
        self.company_info = stock.info
        return self.company_info

    def get_financial_statements(self):
        stock = yf.Ticker(self.ticker)
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cash_flow
        return balance_sheet, income_stmt, cash_flow

    def get_news(self, days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"https://newsapi.org/v2/everything?q={self.company_name}&from={start_date.date()}&to={end_date.date()}&sortBy=popularity&apiKey={self.config.NEWS_API_KEY}"
        
        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json()
            return news_data['articles']
        else:
            print(f"Error fetching news: {response.status_code}")
            return None

    def get_earnings_call_transcripts(self):
        url = f"https://seekingalpha.com/symbol/{self.ticker}/earnings/transcripts"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            transcripts = soup.find_all('a', class_='transcript-link')
            return [{'title': t.text, 'link': 'https://seekingalpha.com' + t['href']} for t in transcripts]
        else:
            print(f"Error fetching earnings call transcripts: {response.status_code}")
            return None

    def calculate_financial_ratios(self):
        if not self.company_info:
            self.get_company_info()
        
        ratios = {
            'P/E Ratio': self.company_info.get('trailingPE', 'N/A'),
            'Forward P/E': self.company_info.get('forwardPE', 'N/A'),
            'PEG Ratio': self.company_info.get('pegRatio', 'N/A'),
            'Price to Book': self.company_info.get('priceToBook', 'N/A'),
            'Debt to Equity': self.company_info.get('debtToEquity', 'N/A'),
            'Return on Equity': self.company_info.get('returnOnEquity', 'N/A'),
            'Return on Assets': self.company_info.get('returnOnAssets', 'N/A'),
            'Profit Margin': self.company_info.get('profitMargins', 'N/A')
        }
        return ratios

    def calculate_technical_indicators(self):
        if self.stock_data is None:
            self.get_stock_data()
        
        data = self.stock_data.copy()
        data['SMA50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA200'] = ta.trend.sma_indicator(data['Close'], window=200)
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_High'] = bollinger.bollinger_hband()
        data['BB_Low'] = bollinger.bollinger_lband()
        return data

    def gpt4_analysis(self, text, task):
        json_formats = {
            "news_sentiment": '''
            {
                "score": float,  // sentiment score between -1 and 1
                "explanation": "string"  // brief explanation of the sentiment
            }
            ''',
            "earnings_call": '''
            {
                "financial_performance": ["string", "string", ...],  // list of key points about financial performance
                "future_outlook": ["string", "string", ...],  // list of key points about future outlook
                "significant_announcements": ["string", "string", ...]  // list of significant announcements
            }
            ''',
            "market_analysis": '''
            {
                "market_conditions": ["string", "string", ...],  // list of key points about current market conditions
                "future_outlook": ["string", "string", ...]  // list of key points about future outlook
            }
            '''
        }

        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a financial analyst assistant. Your task is to {task}. Provide your response in the following JSON format:\n{json_formats[task]}"},
                    {"role": "user", "content": text}
                ]
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error in GPT-4 analysis: {e}")
            return None

    def analyze_news_sentiment(self, news):
        def process_article(article):
            title = article['title']
            description = article['description']
            content = f"Title: {title}\nDescription: {description}"
            sentiment = self.gpt4_analysis(content, "news_sentiment")
            return sentiment

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_article = {executor.submit(process_article, article): article for article in news[:10]}
            sentiments = []
            for future in as_completed(future_to_article):
                sentiment = future.result()
                if sentiment:
                    sentiments.append(sentiment)
        
        return sentiments

    def analyze_earnings_call(self, transcript_link):
        response = requests.get(transcript_link)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            transcript_text = soup.find('div', class_='sa-art').get_text()
            analysis = self.gpt4_analysis(transcript_text[:4000], "earnings_call")
            return analysis
        else:
            print(f"Error fetching transcript: {response.status_code}")
            return None

    def get_competitor_data(self):
        data = {}
        for company in [self.ticker] + self.competitors:
            stock = yf.Ticker(company)
            info = stock.info
            data[company] = {
                'Market Cap': info.get('marketCap', 'N/A'),
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Revenue Growth': info.get('revenueGrowth', 'N/A'),
                'Profit Margin': info.get('profitMargins', 'N/A')
            }
        return pd.DataFrame(data).T

    def plot_stock_price_with_indicators(self, data):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='Close Price')
        plt.plot(data.index, data['SMA50'], label='50-day SMA')
        plt.plot(data.index, data['SMA200'], label='200-day SMA')
        plt.fill_between(data.index, data['BB_High'], data['BB_Low'], alpha=0.1)
        plt.title(f'{self.ticker} Stock Price with Technical Indicators')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(data.index, data['RSI'])
        plt.title('Relative Strength Index (RSI)')
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_technical_analysis.png')
        plt.close()

    def perform_market_analysis(self):
        return self.gpt4_analysis(
            f"Analyze the current market conditions for {self.company_name} and its industry, "
            f"considering its competitors: {', '.join(self.competitors)}.",
            "market_analysis"
        )

    def json_serial(self, obj):
        """JSON serializer for objects not serializable by default JSON code"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            obj = obj.copy()
            obj.index = obj.index.map(str)
            obj.columns = obj.columns.map(str)
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            obj = obj.copy()
            obj.index = obj.index.map(str)
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.floating, np.ndarray)):
            return obj.tolist()
        else:
            return str(obj)

    def save_analysis_to_database(self):
        analysis_data = {
            "company": {
                "name": self.company_name,
                "ticker": self.ticker
            },
            "analysis_date": datetime.now().isoformat(),
            "competitors": self.competitors,
            "financial_data": {
                "ratios": self.financial_ratios,
                "statements": None  # We will serialize this separately
            },
            "technical_indicators": None,  # We will serialize this separately
            "news_sentiment": self.news_sentiment,
            "earnings_call_analysis": self.earnings_call_analysis,
            "competitor_comparison": None,  # We will serialize this separately
            "market_analysis": self.market_analysis
        }

        # Serialize large data separately to avoid exceeding SQLite's storage limits
        financial_statements_json = json.dumps(self.financial_statements, default=self.json_serial)
        technical_indicators_json = self.technical_indicators.to_json(date_format='iso')
        competitor_comparison_json = self.competitor_data.to_json()

        # Store smaller data directly
        analysis_data_json = json.dumps(analysis_data, default=self.json_serial)

        # Insert data into the database
        self.cursor.execute('''
            INSERT OR REPLACE INTO stock_analysis (
                ticker,
                company_name,
                analysis_date,
                competitors,
                financial_ratios,
                financial_statements,
                technical_indicators,
                news_sentiment,
                earnings_call_analysis,
                competitor_comparison,
                market_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.ticker,
            self.company_name,
            analysis_data['analysis_date'],
            json.dumps(self.competitors),
            json.dumps(self.financial_ratios),
            financial_statements_json,
            technical_indicators_json,
            json.dumps(self.news_sentiment),
            json.dumps(self.earnings_call_analysis),
            competitor_comparison_json,
            json.dumps(self.market_analysis)
        ))
        self.conn.commit()
        print(f"Analysis data for {self.ticker} saved to the database.")

    def run_analysis(self):
        print(f"Analyzing {self.company_name} ({self.ticker})")
        self.fetch_all_data()

        if not self.data_exists():
            self.save_analysis_to_database()
        else:
            print(f"Data for {self.ticker} already exists in the database.")

        # Optional: Close the database connection if done
        # self.conn.close()

# Helper function to process multiple tickers
def analyze_all_tickers():
    service = ticker_service()
    markets = ['S&P 500'] #FTSE 100
    stock_dict = {ticker: name for market in markets for ticker, name in service.base_tickers_store[market].items()}
    for ticker, company_name in stock_dict.items():
        try:
            analyzer = StockAnalyzer(ticker, company_name)
            analyzer.run_analysis()
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    analyze_all_tickers()