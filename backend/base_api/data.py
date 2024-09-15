import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple
from config import config
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import random
import concurrent.futures
from collections import deque
import time
import os
 
from playwright.sync_api import sync_playwright, ConsoleMessage, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

 
 

class DataFetcher:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config()
        self.proxies = self._load_proxies()
        self.session = requests.Session()
        self.proxy_queue = deque(self.proxies+[None])  # None represents your own IP
        self.current_proxy = None

    def _get_next_proxy(self) -> str:
        self.current_proxy = self.proxy_queue[0]
        self.proxy_queue.rotate(-1)  # Move the first element to the end
        return self.current_proxy

    def _make_request(self, url: str, headers: Dict = None) -> requests.Response:
        proxy = self._get_next_proxy()
        print('serving request from', 'your IP' if proxy is None else proxy)
        print(url)
        try:
            if proxy is None:
                response = requests.get(url, headers=headers, timeout=10)
            else:
                response = requests.get(url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
            
            response.raise_for_status()
            print(response.status_code, response.reason)
            print(response)
            return response
        except requests.RequestException as e:
            self.logger.error(f"Error making request: {e}")
            return None
        
 
 
 

    def create_session_with_proxy(self) -> str:
        """Create a Browserbase session with proxy configuration."""
        sessions_url = "https://www.browserbase.com/v1/sessions"
        headers = {
            "Content-Type": "application/json",
            "x-bb-api-key": self.config.BROWSERBAE_API_KEY,
        }
        json = {
            "projectId": self.config.BROWSERBAE_PROJECT_ID,
            "proxies": True,
        }

        response = requests.post(sessions_url, json=json, headers=headers)
        response.raise_for_status()
        return response.json()["id"]

    class SolveState:
        """A simple class to track the state of the CAPTCHA solution."""
        started = False
        finished = False

        START_MSG = "browserbase-solving-started"
        END_MSG = "browserbase-solving-finished"

        def handle_console(self, msg: ConsoleMessage) -> None:
            if msg.text == self.START_MSG:
                self.started = True
                print("AI has started solving the CAPTCHA...")
                return

            if msg.text == self.END_MSG:
                self.finished = True
                print("AI solved the CAPTCHA!")
                return

    def _make_request_playwright(self, url: str, wait_for_selector: str = None) -> str:
        print(f'Serving request from Browserbase proxy')
        print(url)

        def run(page: Page):
            state = self.SolveState()
            page.on("console", state.handle_console)

            try:
                page.goto(url)

                try:
                    with page.expect_console_message(
                        lambda msg: msg.text == self.SolveState.END_MSG,
                        timeout=10000,
                    ):
                        pass
                except PlaywrightTimeoutError:
                    print("Timeout: No CAPTCHA solving event detected after 10 seconds")
                    print(f"Solve state: {state.started=} {state.finished=}")

                if state.started != state.finished:
                    raise Exception(f"Solving mismatch! {state.started=} {state.finished=}")

                if state.started == state.finished == False:
                    print("No CAPTCHA was presented, or was solved too quick to see.")
                else:
                    print("CAPTCHA is complete.")

                if wait_for_selector:
                    page.wait_for_selector(wait_for_selector)
                else:
                    page.locator("body").wait_for(state="visible")

                content = page.content()
                print(f"{page.url=}")
                print(f"{page.title()=}")
                return content
            except Exception as e:
                self.logger.error(f"Error making Playwright request: {e}")
                return None

        with sync_playwright() as playwright:
            session_id = self.create_session_with_proxy()
            browser = playwright.chromium.connect_over_cdp(
                f"wss://connect.browserbase.com?apiKey={self.config.BROWSERBAE_API_KEY}&sessionId={session_id}"
            )

            print(
                "Connected to Browserbase.",
                f"{browser.browser_type.name} version {browser.version}",
            )

            context = browser.contexts[0]
            page = context.pages[0]

            try:
                return run(page)
            finally:
                page.close()
                browser.close()
    def _load_proxies(self) -> List[str]:
        # Load proxies from a file or API
        # For this example, we'll use a placeholder list
        return [
            "http://143.198.189.246:9000",
            "http://165.227.201.152:9000"
        ]

    def test_proxies(self, test_url: str = "http://httpbin.org/ip") -> Dict[str, bool]:
        """
        Test all proxies and return a dictionary of working proxies.
        
        :param test_url: URL to test the proxies against (default: http://httpbin.org/ip)
        :return: Dictionary with proxy URLs as keys and boolean values indicating if they work
        """
        working_proxies = {}

        def test_proxy(proxy):
            try:
                response = requests.get(test_url, proxies={"http": proxy, "https": proxy}, timeout=10)
                if response.status_code == 200:
                    self.logger.info(f"Proxy {proxy} is working")
                    return proxy, True
                else:
                    self.logger.warning(f"Proxy {proxy} returned status code {response.status_code}")
                    return proxy, False
            except requests.RequestException as e:
                self.logger.error(f"Error testing proxy {proxy}: {e}")
                return proxy, False

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_proxy = {executor.submit(test_proxy, proxy): proxy for proxy in self.proxies}
            for future in concurrent.futures.as_completed(future_to_proxy):
                proxy, is_working = future.result()
                working_proxies[proxy] = is_working

        # Update the list of proxies to only include working ones
        self.proxies = [proxy for proxy, is_working in working_proxies.items() if is_working]
        print(self.proxies)

        if not self.proxies:
            self.logger.critical("No working proxies found!")

        return working_proxies
    def _get_random_proxy(self) -> str:
        return random.choice(self.proxies)
 


    def get_news(self, company_name: str, days: int = 7) -> List[Dict]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={company_name}&"
            f"from={start_date.date()}&"
            f"to={end_date.date()}&"
            f"sortBy=popularity&"
            f"apiKey={self.config.NEWS_API_KEY}"
        )
        response = self._make_request(url)
        if response.status_code == 200:
            news_data = response.json()
            articles = news_data.get('articles', [])
            self.logger.info(f"Fetched {len(articles)} news articles for {company_name}.")
            return articles
        else:
            self.logger.error(f"Error fetching news: {response.status_code}")
            return []

    def get_earnings_call_transcripts(self, ticker: str) -> List[Dict]:
        url = f"https://seekingalpha.com/symbol/{ticker}/earnings/transcripts"
        print(url)
        headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'max-age=0',
            'cookie': 'machine_cookie=7507411255212; session_id=31c5fa6b-ef9d-4c82-b040-739eaa4179d9; LAST_VISITED_PAGE=%7B%22pathname%22%3A%22https%3A%2F%2Fseekingalpha.com%2Fsymbol%2FAAPL%2Fearnings%2Ftranscripts%22%2C%22pageKey%22%3A%22d77496d7-5380-4108-983e-9459068f5213%22%7D; _sasource=; _gcl_au=1.1.1931947578.1726357346; sailthru_pageviews=1; _ga_KGRFF2R2C5=GS1.1.1726357346.1.0.1726357346.60.0.0; _ga=GA1.1.1661704173.1726357346; _fbp=fb.1.1726357346372.15562469244440192; sailthru_content=aecf9e6d8252c37a01aed4a287b07535; sailthru_visitor=1f5f2dba-7852-44f6-841c-dafb7bf39547; _pcid=%7B%22browserId%22%3A%22m12sla4gt994glfe%22%7D; __tbc=%7Bkpex%7D8uF_o9kL1K0D_Jajq9J2OUEIDWYxZpHwmOI6jzizNnMwA75AfFmA_eSFgY7p3f_X; __pat=-14400000; __pvi=eyJpZCI6InYtMjAyNC0wOS0xNC0xOS00Mi0yNi05NDYtenZheEt0SHNOZHNBeTR6dy1lOTVkNmFhN2QwOWY2ZTMxZTA4ZGExMzM1NTU2NmVmYyIsImRvbWFpbiI6Ii5zZWVraW5nYWxwaGEuY29tIiwidGltZSI6MTcyNjM1NzM0NzEwNn0%3D; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAE0RXQF8g; xbc=%7Bkpex%7DGiqhUXofateh_SljSBi3iw; __hstc=234155329.4c11a09336102313a1ca2ca822cff2b6.1726357347334.1726357347334.1726357347334.1; hubspotutk=4c11a09336102313a1ca2ca822cff2b6; __hssrc=1; __hssc=234155329.1.1726357347334; pxcts=003e6df6-72f3-11ef-ad9f-66f9af1e480b; _pxvid=003e599e-72f3-11ef-ad9e-88f45339676a; _px3=3639746c35f7b46eb6634980561b25a88cf640e42c902d81c460f3a858760ef9:KfD0EBfRIR//9kcTGPD47TkXEHpMDk6qVnxncQew+DPxPH/Pgj/b79REtkVNlDV3DK9fIsIqpw/BGc8Vb48qvA==:1000:ruaxoM6QeRJ+0p2qP9NbDSBrvTVYZUtowaNZDxMZoIw9DQE4hvOQZmP4f0MaHyHZWN8lSRzwW7emzkQrH8Z9sbZcfyvWZuk64zEzlE8wIcpzOqpvJ6hdtC/nlYsCp7hw1PgCzImmgKzmqxiuYf9/47cHwhxNlSxYmbTIRhYyrMzbNUzWPchrC48gpDS2QsRoWM4usqkzKQeOWrWelz/rBF9ENZ2Xhci/Bpq2QErOnzw=; _pxde=9a13a9ed1776a4cbbd3e4be2ade4844dcfe2c7896a450f61ac40105933fe8720:eyJ0aW1lc3RhbXAiOjE3MjYzNTczNDg2NTksImZfa2IiOjB9; userLocalData_mone_session_lastSession=%7B%22machineCookie%22%3A%227507411255212%22%2C%22machineCookieSessionId%22%3A%227507411255212%261726357344854%22%2C%22sessionStart%22%3A1726357344854%2C%22sessionEnd%22%3A1726359160530%2C%22firstSessionPageKey%22%3A%22d77496d7-5380-4108-983e-9459068f5213%22%2C%22isSessionStart%22%3Afalse%2C%22lastEvent%22%3A%7B%22event_type%22%3A%22mousemove%22%2C%22timestamp%22%3A1726357360530%7D%7D',
            'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        }
        #response = self._make_request(url,headers)
        response = self._make_request_playwright(url)

        # time.sleep(3)
    
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            transcripts = soup.find_all('a', {'data-test-id': 'post-list-item-title'})
     
            transcripts_data = []
            for t in transcripts:
                transcript_url = 'https://seekingalpha.com' + t['href']
                try:
                    transcript_content = self.scrape_transcript_content(transcript_url)
                    transcripts_data.append({
                        'title': t.text.strip(),
                        'link': transcript_url,
                        'content': transcript_content
                    })
                    # time.sleep(3)
                except Exception as e:
                    self.logger.error(f"Error fetching transcript content for {transcript_url}: {str(e)}")

            self.logger.info(f"Fetched {len(transcripts_data)} earnings call transcripts for {ticker}.")
            return transcripts_data
        else:
            self.logger.error(f"Error fetching earnings call transcripts: {response.status_code}")
            return []

    def scrape_transcript_content(self, url: str) -> str:
        headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'max-age=0',
            'cookie': 'machine_cookie=7507411255212; session_id=31c5fa6b-ef9d-4c82-b040-739eaa4179d9; LAST_VISITED_PAGE=%7B%22pathname%22%3A%22https%3A%2F%2Fseekingalpha.com%2Fsymbol%2FAAPL%2Fearnings%2Ftranscripts%22%2C%22pageKey%22%3A%22d77496d7-5380-4108-983e-9459068f5213%22%7D; _sasource=; _gcl_au=1.1.1931947578.1726357346; sailthru_pageviews=1; _ga_KGRFF2R2C5=GS1.1.1726357346.1.0.1726357346.60.0.0; _ga=GA1.1.1661704173.1726357346; _fbp=fb.1.1726357346372.15562469244440192; sailthru_content=aecf9e6d8252c37a01aed4a287b07535; sailthru_visitor=1f5f2dba-7852-44f6-841c-dafb7bf39547; _pcid=%7B%22browserId%22%3A%22m12sla4gt994glfe%22%7D; __tbc=%7Bkpex%7D8uF_o9kL1K0D_Jajq9J2OUEIDWYxZpHwmOI6jzizNnMwA75AfFmA_eSFgY7p3f_X; __pat=-14400000; __pvi=eyJpZCI6InYtMjAyNC0wOS0xNC0xOS00Mi0yNi05NDYtenZheEt0SHNOZHNBeTR6dy1lOTVkNmFhN2QwOWY2ZTMxZTA4ZGExMzM1NTU2NmVmYyIsImRvbWFpbiI6Ii5zZWVraW5nYWxwaGEuY29tIiwidGltZSI6MTcyNjM1NzM0NzEwNn0%3D; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAE0RXQF8g; xbc=%7Bkpex%7DGiqhUXofateh_SljSBi3iw; __hstc=234155329.4c11a09336102313a1ca2ca822cff2b6.1726357347334.1726357347334.1726357347334.1; hubspotutk=4c11a09336102313a1ca2ca822cff2b6; __hssrc=1; __hssc=234155329.1.1726357347334; pxcts=003e6df6-72f3-11ef-ad9f-66f9af1e480b; _pxvid=003e599e-72f3-11ef-ad9e-88f45339676a; _px3=3639746c35f7b46eb6634980561b25a88cf640e42c902d81c460f3a858760ef9:KfD0EBfRIR//9kcTGPD47TkXEHpMDk6qVnxncQew+DPxPH/Pgj/b79REtkVNlDV3DK9fIsIqpw/BGc8Vb48qvA==:1000:ruaxoM6QeRJ+0p2qP9NbDSBrvTVYZUtowaNZDxMZoIw9DQE4hvOQZmP4f0MaHyHZWN8lSRzwW7emzkQrH8Z9sbZcfyvWZuk64zEzlE8wIcpzOqpvJ6hdtC/nlYsCp7hw1PgCzImmgKzmqxiuYf9/47cHwhxNlSxYmbTIRhYyrMzbNUzWPchrC48gpDS2QsRoWM4usqkzKQeOWrWelz/rBF9ENZ2Xhci/Bpq2QErOnzw=; _pxde=9a13a9ed1776a4cbbd3e4be2ade4844dcfe2c7896a450f61ac40105933fe8720:eyJ0aW1lc3RhbXAiOjE3MjYzNTczNDg2NTksImZfa2IiOjB9; userLocalData_mone_session_lastSession=%7B%22machineCookie%22%3A%227507411255212%22%2C%22machineCookieSessionId%22%3A%227507411255212%261726357344854%22%2C%22sessionStart%22%3A1726357344854%2C%22sessionEnd%22%3A1726359160530%2C%22firstSessionPageKey%22%3A%22d77496d7-5380-4108-983e-9459068f5213%22%2C%22isSessionStart%22%3Afalse%2C%22lastEvent%22%3A%7B%22event_type%22%3A%22mousemove%22%2C%22timestamp%22%3A1726357360530%7D%7D',
            'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        }
        # response = self._make_request_playwright(url,headers)
        response = self._make_request_playwright(url, wait_for_selector='div[data-test-id="content-container"]')


        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            content_div = soup.find('div', {'data-test-id': 'content-container'})
            
            if content_div:
 
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.text.strip() for p in paragraphs])
                return content
        self.logger.error(f"Error scraping transcript content: {response.status_code}")
        return ""

    def get_financials(self, ticker: str) -> Dict:
        """Fetch company financial data."""
        url = (
            f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
            f"?limit=1&apikey={self.config.FMP_API_KEY}"
        )
 
        response = self._make_request(url)
        if response.status_code == 200:
            data = response.json()
            financials = data[0] if data else {}
            self.logger.info(f"Fetched financial data for {ticker}.")
            return financials
        else:
            self.logger.error(f"Error fetching financials: {response.status_code}")
            return {}
    
    def get_company_profile(self, ticker: str) -> Dict:
        """Fetch company profile data."""
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={self.config.FMP_API_KEY}"
        print(url)
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            profile = data[0] if data else {}
            self.logger.info(f"Fetched company profile for {ticker}.")
            return profile
        else:
            self.logger.error(f"Error fetching company profile: {response.status_code}")
            return {}

    def get_industry_reports(self, industry: str) -> List[Dict]:
        """Fetch industry reports."""
        self.logger.info(f"Fetching industry reports for {industry}.")
        url = f"https://financialmodelingprep.com/api/v3/stock-screener?industry={industry}&limit=100&apikey={self.config.FMP_API_KEY}"
        print(url)
        response = requests.get(url)
        if response.status_code == 200:
            companies = response.json()
            market_caps = [company.get('marketCap', 0) for company in companies]
            total_market_cap = sum(market_caps)
            average_market_cap = total_market_cap / len(market_caps) if companies else 0
            if len(market_caps)==0:
                    market_caps = [0]
            market_cap_info = ""
            if market_caps:
              
                market_cap_info = f"The largest company has a market cap of ${max(market_caps):,}, " \
                                  f"while the smallest has a market cap of ${min(market_caps):,}. "

            prompt = f"""Generate a brief industry report for the {industry} industry.
            {market_cap_info}
            Include key trends, major players, and recent developments."""
 
            reports = [
                {
                    'title': f"{industry} Industry Report",
                    'content': (
                        f"The total market capitalization of the {industry} industry is ${total_market_cap:,}. "
                        f"The average market capitalization is ${average_market_cap:,}. "
                        f"There are {len(companies)} companies in this industry. "
                        f"The largest company has a market cap of ${max(market_caps):,}, "
                        f"while the smallest has a market cap of ${min(market_caps):,}. "
                        f"The median market cap is ${sorted(market_caps)[len(market_caps)//2]:,}."
                    ),
                },
                {
                    'title': f"{industry} Industry Concentration",
                    'content': self.calculate_industry_concentration(market_caps),
                },
                
                {
                    'title': f"{industry} Growth Analysis",
                    'content': self.analyze_industry_growth(companies) if len(companies) > 0 else ''
                },
                
                
                
                
            ]
            self.logger.info(f"Fetched industry reports for {industry}.")
            return reports
        else:
            self.logger.error(f"Error fetching industry reports: {response.status_code}")
            return []

 


    def calculate_industry_concentration(self, market_caps: List[float]) -> str:
        sorted_caps = sorted(market_caps, reverse=True)
        if sum(market_caps) ==0:
            market_caps = [1]
        top_5_share = sum(sorted_caps[:5]) / sum(market_caps) * 100
        return (f"The top 5 companies account for {top_5_share:.2f}% of the industry's market cap, "
                f"indicating a {'highly' if top_5_share > 60 else 'moderately' if top_5_share > 40 else 'less'} concentrated market.")

    def analyze_industry_growth(self, companies: List[Dict]) -> str:
    
        total_market_cap = sum(company.get('marketCap', 0) for company in companies)
        
        
        avg_market_cap = total_market_cap / len(companies) if companies else 0
        
        
        largest_company = max(companies, key=lambda x: x.get('marketCap', 0))
        smallest_company = min(companies, key=lambda x: x.get('marketCap', 0))
        
        return (f"The total market cap of the industry is ${total_market_cap:,.0f}. "
                f"The average market cap is ${avg_market_cap:,.0f}. "
                f"The largest company is {largest_company['companyName']} with a market cap of ${largest_company['marketCap']:,.0f}. "
                f"The smallest company is {smallest_company['companyName']} with a market cap of ${smallest_company['marketCap']:,.0f}.")



    def analyze_financial_health(self, companies: List[Dict]) -> str:
        debt_to_equity_ratios = [company.get('debtToEquityRatio', 0) for company in companies if company.get('debtToEquityRatio') is not None]
        avg_debt_to_equity = sum(debt_to_equity_ratios) / len(debt_to_equity_ratios) if debt_to_equity_ratios else 0
        return f"The average debt-to-equity ratio in the industry is {avg_debt_to_equity:.2f}, indicating {'high' if avg_debt_to_equity > 2 else 'moderate' if avg_debt_to_equity > 1 else 'low'} leverage."
    
    def get_yfinance_data(self, ticker: str) -> Dict:
        """Fetch comprehensive data from yfinance."""
        stock = yf.Ticker(ticker)
        data = {
            'info': stock.info,
            #'history': self._convert_df_to_dict(stock.history(period="5y")),
            'actions': self._convert_df_to_dict(stock.actions),
            'dividends': self._convert_df_to_dict(stock.dividends),
            'splits': self._convert_df_to_dict(stock.splits),
            'financials': {
                'income_stmt': self._convert_df_to_dict(stock.income_stmt),
                'balance_sheet': self._convert_df_to_dict(stock.balance_sheet),
                'cashflow': self._convert_df_to_dict(stock.cashflow),
            },
            'holders': {
                'major_holders': stock.major_holders,
                'institutional_holders': self._convert_df_to_dict(stock.institutional_holders) if hasattr(stock.institutional_holders, 'to_dict') else None,
                'mutualfund_holders': self._convert_df_to_dict(stock.mutualfund_holders) if hasattr(stock.mutualfund_holders, 'to_dict') else None,
            },
            'sustainability': self._convert_df_to_dict(stock.sustainability) if stock.sustainability is not None else None,
            'recommendations': self._convert_df_to_dict(stock.recommendations) if stock.recommendations is not None else None,
            'analyst_price_targets': stock.analyst_price_targets if stock.analyst_price_targets is not None else None,
            'earnings_dates': self._convert_df_to_dict(stock.earnings_dates) if stock.earnings_dates is not None else None,
            'isin': stock.isin,
            'options': stock.options,
            'news': stock.news,
        }
        self.logger.info(f"Fetched comprehensive yfinance data for {ticker}.")
        return data

    def _convert_df_to_dict(self, df):
        """Convert a DataFrame to a dictionary with serializable values."""
        data = pd.DataFrame(df)
        return data.to_dict()

    def _convert_value(self, value):
        """Convert a value to a serializable format."""
        if isinstance(value, dict):
            return {str(k): self._convert_value(v) for k, v in value.items()}
        elif isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        elif isinstance(value, (np.int64, np.float64)):
            return value.item()
        elif pd.isna(value):
            return None
        return value

    def analyze_growth_trends(self, ticker: str) -> Dict:
        """Analyze growth trends across multiple timeframes."""
        stock = yf.Ticker(ticker)
        financials = stock.financials
        
        revenue_growth = self._calculate_growth_rates(financials.loc['Total Revenue'])
        net_income_growth = self._calculate_growth_rates(financials.loc['Net Income'])
        
        return {
            'revenue_growth': revenue_growth,
            'net_income_growth': net_income_growth,
            'analysis': self._interpret_growth_trends(revenue_growth, net_income_growth)
        }

    def _calculate_growth_rates(self, series: pd.Series) -> Dict[str, float]:
        """Calculate growth rates for different timeframes."""
        growth_rates = {}
        for i in range(1, len(series)):
            timeframe = f"{i}Y"
            growth_rate = (series.iloc[0] / series.iloc[i]) ** (1/i) - 1
            growth_rates[timeframe] = growth_rate
        return growth_rates

    def _interpret_growth_trends(self, revenue_growth: Dict[str, float], net_income_growth: Dict[str, float]) -> str:
        """Interpret growth trends and provide insights."""
        analysis = "Growth Trend Analysis:\n"
        
        
        for timeframe in revenue_growth.keys():
            rev_growth = revenue_growth[timeframe]
            inc_growth = net_income_growth[timeframe]
            analysis += f"{timeframe} Growth: Revenue {rev_growth:.2%}, Net Income {inc_growth:.2%}\n"
            
            if rev_growth > inc_growth:
                analysis += "  - Revenue growing faster than net income, potential margin pressure.\n"
            elif inc_growth > rev_growth:
                analysis += "  - Net income growing faster than revenue, indicating improving efficiency.\n"
        
        
        rev_trend = self._analyze_trend_direction(revenue_growth)
        inc_trend = self._analyze_trend_direction(net_income_growth)
        
        analysis += f"Revenue Trend: {rev_trend}\n"
        analysis += f"Net Income Trend: {inc_trend}\n"
        
        return analysis

    def _analyze_trend_direction(self, growth_rates: Dict[str, float]) -> str:
        """Analyze the direction of a growth trend."""
        values = list(growth_rates.values())
        if all(x > y for x, y in zip(values, values[1:])):
            return "Accelerating growth"
        elif all(x < y for x, y in zip(values, values[1:])):
            return "Decelerating growth"
        else:
            return "Mixed growth pattern"

    def analyze_financial_health(self, ticker: str) -> Dict:
        """Analyze financial health using various ratios and metrics."""
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        
        
        def safe_get(df, key):
            possible_keys = [key, key.lower(), key.replace(' ', '')]
            for k in possible_keys:
                if k in df.index:
                    return df.loc[k]
            return None

        
        current_assets = safe_get(balance_sheet, 'Total Current Assets')
        current_liabilities = safe_get(balance_sheet, 'Total Current Liabilities')
        total_liabilities = safe_get(balance_sheet, 'Total Liab')
        stockholder_equity = safe_get(balance_sheet, 'Total Stockholder Equity')
        net_income = safe_get(income_stmt, 'Net Income')

        current_ratio = current_assets / current_liabilities if current_assets is not None and current_liabilities is not None else None
        debt_to_equity = total_liabilities / stockholder_equity if total_liabilities is not None and stockholder_equity is not None else None
        roe = net_income / stockholder_equity if net_income is not None and stockholder_equity is not None else None

        
        current_ratio = float(current_ratio.iloc[0]) if current_ratio is not None else None
        debt_to_equity = float(debt_to_equity.iloc[0]) if debt_to_equity is not None else None
        roe = float(roe.iloc[0]) if roe is not None else None

        return {
            'current_ratio': current_ratio,
            'debt_to_equity': debt_to_equity,
            'return_on_equity': roe,
            'analysis': self._interpret_financial_health(current_ratio, debt_to_equity, roe)
        }

    def _interpret_financial_health(self, current_ratio: float, debt_to_equity: float, roe: float) -> str:
        """Interpret financial health metrics and provide insights."""
        analysis = "Financial Health Analysis:\n"
        
        if current_ratio is not None:
            if current_ratio > 2:
                analysis += "- Strong liquidity position, potentially underutilizing assets.\n"
            elif current_ratio < 1:
                analysis += "- Potential liquidity issues, may struggle to meet short-term obligations.\n"
            else:
                analysis += "- Adequate liquidity position.\n"
        else:
            analysis += "- Current ratio data not available.\n"
        
        if debt_to_equity is not None:
            if debt_to_equity > 2:
                analysis += "- High leverage, increased financial risk.\n"
            elif debt_to_equity < 0.5:
                analysis += "- Conservative capital structure, may be underutilizing leverage.\n"
            else:
                analysis += "- Balanced capital structure.\n"
        else:
            analysis += "- Debt-to-equity ratio data not available.\n"
        
        if roe is not None:
            if roe > 0.2:
                analysis += "- Strong return on equity, efficient use of shareholder funds.\n"
            elif roe < 0.05:
                analysis += "- Low return on equity, may indicate inefficient use of capital.\n"
            else:
                analysis += "- Moderate return on equity.\n"
        else:
            analysis += "- Return on equity data not available.\n"
        
        return analysis

    def analyze_market_sentiment(self, ticker: str) -> Dict:
        """Analyze market sentiment using price trends, volatility, and analyst recommendations."""
        stock = yf.Ticker(ticker)
        history = stock.history(period="1y")
        recommendations = stock.recommendations
        
        price_trend = self._calculate_price_trend(history)
        volatility = history['Close'].pct_change().std() * (252 ** 0.5)  
        analyst_sentiment = self._analyze_analyst_recommendations(recommendations)
        
        return {
            'price_trend': price_trend,
            'volatility': volatility,
            'analyst_sentiment': analyst_sentiment,
            'analysis': self._interpret_market_sentiment(price_trend, volatility, analyst_sentiment)
        }

    def _calculate_price_trend(self, history: pd.DataFrame) -> float:
        """Calculate price trend using linear regression."""
        x = np.arange(len(history))
        y = history['Close'].values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    def _analyze_analyst_recommendations(self, recommendations: pd.DataFrame) -> Dict[str, int]:
        """Analyze recent analyst recommendations."""
        if recommendations is None or recommendations.empty:
            return {"No Data": 0}
        
        recent_recommendations = recommendations.iloc[-10:]  
        
        
        possible_columns = ['To Grade', 'Action', 'Recommendation']
        recommendation_column = next((col for col in possible_columns if col in recent_recommendations.columns), None)
        
        if recommendation_column is None:
            return {"No Valid Recommendation Column": 0}
        
        return recent_recommendations[recommendation_column].value_counts().to_dict()

    def _interpret_market_sentiment(self, price_trend: float, volatility: float, analyst_sentiment: Dict[str, int]) -> str:
        """Interpret market sentiment metrics and provide insights."""
        analysis = "Market Sentiment Analysis:\n"
        
        if price_trend > 0:
            analysis += f"- Positive price trend (slope: {price_trend:.4f}), indicating bullish sentiment.\n"
        else:
            analysis += f"- Negative price trend (slope: {price_trend:.4f}), indicating bearish sentiment.\n"
        
        analysis += f"- Annualized volatility: {volatility:.2%}\n"
        if volatility > 0.4:
            analysis += "  High volatility, suggesting increased uncertainty.\n"
        elif volatility < 0.2:
            analysis += "  Low volatility, suggesting market stability.\n"
        
        if analyst_sentiment:
            analysis += "- Recent analyst recommendations:\n"
            for grade, count in analyst_sentiment.items():
                analysis += f"  {grade}: {count}\n"
        else:
            analysis += "- No recent analyst recommendations available.\n"
        
        return analysis

    def get_comprehensive_analysis(self, ticker: str) -> Dict:
        """Perform a comprehensive analysis of a stock."""
        analysis_results = {}
        
        try:
            analysis_results['growth_analysis'] = self.analyze_growth_trends(ticker)
        except Exception as e:
            self.logger.error(f"Error in growth analysis for {ticker}: {str(e)}")

        try:
            analysis_results['financial_health'] = self.analyze_financial_health(ticker)
        except Exception as e:
            self.logger.error(f"Error in financial health analysis for {ticker}: {str(e)}")

        try:
            analysis_results['market_sentiment'] = self.analyze_market_sentiment(ticker)
        except Exception as e:
            self.logger.error(f"Error in market sentiment analysis for {ticker}: {str(e)}")

        analysis_results['summary'] = self._generate_summary(
            analysis_results.get('growth_analysis', {}),
            analysis_results.get('financial_health', {}),
            analysis_results.get('market_sentiment', {})
        )
        
        return analysis_results

    def _generate_summary(self, growth_analysis: Dict, financial_health: Dict, market_sentiment: Dict) -> str:
        """Generate a summary of the comprehensive analysis."""
        summary = "Comprehensive Analysis Summary:\n\n"
        
        if 'analysis' in growth_analysis:
            summary += "Growth Trends:\n"
            summary += growth_analysis['analysis'] + "\n\n"
        
        if 'analysis' in financial_health:
            summary += "Financial Health:\n"
            summary += financial_health['analysis'] + "\n\n"
        
        if 'analysis' in market_sentiment:
            summary += "Market Sentiment:\n"
            summary += market_sentiment['analysis'] + "\n\n"
        
        return summary


# if __name__ == "__main__":
#     fetcher = DataFetcher()
#     # working_proxies = fetcher.test_proxies()
#     # print("Working proxies:", working_proxies)
#     ticker = "AAPL"
#     comp = fetcher.get_yfinance_data(ticker)
#     print(comp)
#     # # comp = fetcher.get_company_profile(ticker)
    
#     # analysis = fetcher.get_industry_reports(comp['industry'])
#     # print(analysis)

