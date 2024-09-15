import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
from data import DataFetcher
from macros import ticker_service
import json
import logging
from openai import OpenAI
from config import config
import os
import pickle
from functools import lru_cache
from openai import OpenAI
import pyaudio, wave, keyboard, faster_whisper, torch.cuda, os
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import pandas as pd
from datetime import datetime
import torch
import asyncio
from openai import OpenAI,AsyncOpenAI
from config import config
import faster_whisper
import elevenlabs
from chart import ContextBasedChartGenerator
system_prompt = """"
You  are a associate at a finicial firm.integrated with a sophisticated GraphRAG (Retrieval Augmented Generation) system for financial market analysis. Your responses are based on the context provided in each user query, which includes relevant information retrieved from the GraphRAG system. Your role is to interpret this context and provide insightful, impressive, and relatable responses for both technical and non-technical audiences.
 Your responses should be short, very short, insanely crazy insights and analytics, things that have a wow fucking factor, that will be so helpful, make it relatable, sound very very natural
 
in the users query, automatically fix typos or context, 
Key points to remember:
ensure to make sure the user get reallly ususally key insight
1. Always base your responses on the context provided. Do not invent or assume additional information.

2. Maintain a professional yet engaging tone, like a top Wall Street analyst with a gift for clear communication, and precise puzzle solving and analysis
3. its a conversation setting, you don't wanna bore or talk to much, remember to give short references, exact sources and timeframes. because i want the listeners to know its valid and not just making stuff up.
never act unsure, be 100% confidence. need you to sound very very captivatiing, realistic still, very very british also, and  avoid sound robotic

do not add any special characters, like **
"""


class GraphRAG:
    def __init__(self):
        self.graph = nx.Graph()
        self.data_fetcher = DataFetcher()
        self.ticker_service = ticker_service()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.logger = self.setup_logger()
        self.graph_file = "graph.pickle"
        self.embeddings_file = "embeddings.pickle"
        self.json_graph_file = "graph_data.json"
        self.embeddings = {}
        self.progress_file = "graph_progress.json"
        self.companies_processed = self.load_progress()

    def setup_logger(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler("graph_rag.log")
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return []

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.companies_processed, f, default=self.json_serial)

    @staticmethod
    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    def build_graph(self):
        self.logger.info("Starting to build graph")
        if os.path.exists(self.graph_file) and os.path.exists(self.embeddings_file):
            self.logger.info("Loading existing graph")
            self.load_graph()
        else:
            self.logger.info("Building new graph")
            tickers = self.ticker_service.base_tickers_store
            first_exchange = next(iter(tickers))
            temp = {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corporation",
                "GOOGL": "Alphabet Inc.",
                "AMZN": "Amazon.com, Inc.",
                "TSLA": "Tesla, Inc.",
                
                "NVDA": "NVIDIA Corporation",
                "JNJ": "Johnson & Johnson",
                "V": "Visa Inc.",
        
 
                "SBUX": "Starbucks Corporation",
            
                "JPM": "JPMorgan Chase & Co.",
      
                "UNH": "UnitedHealth Group Incorporated",
                "HD": "The Home Depot, Inc.",
                "MA": "Mastercard Incorporated",
                "PFE": "Pfizer Inc.",
                "KO": "The Coca-Cola Company",
                "DIS": "The Walt Disney Company",
                "CVX": "Chevron Corporation",
                "PEP": "PepsiCo, Inc.",
         
                "BAC": "Bank of America Corporation",
                "META": "Meta Platforms, Inc.",
                "CSCO": "Cisco Systems, Inc.",
                "INTC": "Intel Corporation",
                "MRK": "Merck & Co., Inc.",
                "ABT": "Abbott Laboratories",
                "ORCL": "Oracle Corporation",
                "WMT": "Walmart Inc.",
             
                "COST": "Costco Wholesale Corporation",
                "NKE": "NIKE, Inc.",
                "AMD": "Advanced Micro Devices, Inc.",
           
                "UPS": "United Parcel Service, Inc.",
                "TXN": "Texas Instruments Incorporated",
           
            }

            companies = list(temp.items())

            for company in companies:
                ticker, company_name = company
                if ticker not in self.companies_processed:
                    try:
                        try:
                            self._add_company_data(ticker, company_name)
                            self.companies_processed.append(ticker)
                        except Exception as e:
                            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                            continue
                        self.save_progress()
                        self.save_graph()  # Save after each company is processed
                    except Exception as e:
                        self.logger.error(f"Error processing {ticker}: {str(e)}")
                        # Save progress and graph before exiting
                        self.save_progress()
                        self.save_graph()
                        raise  # Re-raise the exception to stop the process

        self.save_graph_json()
        self.logger.info("Graph building completed")

    def save_graph(self):
        self.logger.debug("Saving graph to pickle files")
        with open(self.graph_file, "wb") as f:
            pickle.dump(self.graph, f)
        with open(self.embeddings_file, "wb") as f:
            pickle.dump(self.embeddings, f)
        self.logger.info("Graph saved successfully")

    def load_graph(self):
        self.logger.debug("Loading graph from pickle files")
        with open(self.graph_file, "rb") as f:
            self.graph = pickle.load(f)
        with open(self.embeddings_file, "rb") as f:
            self.embeddings = pickle.load(f)
        self.logger.info("Graph loaded successfully")

    def save_graph_json(self):
        self.logger.debug("Saving graph to JSON file")
        graph_data = {
            "nodes": [
                {"id": node, **{k: v for k, v in data.items() if k != "id"}}
                for node, data in self.graph.nodes(data=True)
            ],
            "links": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ],
        }
        with open(self.json_graph_file, "w") as f:
            json.dump(graph_data, f, indent=2)
        self.logger.info("Graph JSON saved successfully")
    
    def _add_company_node(self, ticker: str, company_name: str):
        self.logger.debug(f"Adding company node: {ticker} - {company_name}")
        self.graph.add_node(ticker, name=company_name, type="company", ticker=ticker)
    def add_company_nodes_from_json(self, json_file_path: str):
        self.logger.debug(f"Adding company nodes from JSON file: {json_file_path}")
        try:
            with open(json_file_path, 'r') as file:
                graph_data = json.load(file)
            
            for node in graph_data.get('nodes', []):
                if node.get('type') == 'company':
                    ticker = node.get('ticker')
                    name = node.get('name')
                    if ticker and name:
                        self._add_company_node(ticker, name)
                    else:
                        self.logger.warning(f"Skipping node due to missing data: {node}")
            
            self.logger.info(f"Successfully added company nodes from {json_file_path}")
        except Exception as e:
            self.logger.error(f"Error adding company nodes from JSON: {str(e)}")
    def _add_company_data(self, ticker: str, company_name: str):
        self.logger.debug(f"Adding company data for: {ticker}")
        try:
            self._add_company_node(ticker, company_name)
            news = self.data_fetcher.get_news(company_name)
            transcripts = []  # Assuming this is not used
            financials = self.data_fetcher.get_financials(ticker)
            profile = self.data_fetcher.get_company_profile(ticker)
            yfinance_data = self.data_fetcher.get_yfinance_data(ticker)
            analysis = self.data_fetcher.get_comprehensive_analysis(ticker)

            # Convert any Timestamp objects in the data
            # financials = self.convert_timestamps(financials)
            # profile = self.convert_timestamps(profile)
            # yfinance_data = self.convert_timestamps(yfinance_data)
            # analysis = self.convert_timestamps(analysis)

            for article in news:
                node_id = f"news_{hash(article['title'])}"
                self.graph.add_node(node_id, type="news", **article)
                self.graph.add_edge(ticker, node_id)
                self.embeddings[node_id] = self.model.encode(
                    self._get_node_text(self.graph.nodes[node_id])
                )

            for transcript in transcripts:
                node_id = f"transcript_{hash(transcript['title'])}"
                self.graph.add_node(node_id, type="transcript", **transcript)
                self.graph.add_edge(ticker, node_id)

            if financials:
                node_id = f"financials_{ticker}"
                self.graph.add_node(node_id, type="financials", **financials)
                self.graph.add_edge(ticker, node_id)

                # Add advanced metrics and insights
                self.add_advanced_financial_metrics(ticker, financials)

            if profile:
                node_id = f"profile_{ticker}"
                self.graph.add_node(node_id, type="profile", **profile)
                self.graph.add_edge(ticker, node_id)

            if profile and "industry" in profile:
                industry_reports = self.data_fetcher.get_industry_reports(
                    profile["industry"]
                )
                for report in industry_reports:
                    node_id = f"industry_report_{hash(report['title'])}"
                    self.graph.add_node(node_id, type="industry_report", **report)
                    self.graph.add_edge(ticker, node_id)

            # # Add yfinance data
            # node_id = f"yfinance_{ticker}"
            # self.graph.add_node(node_id, type="yfinance", **yfinance_data)
            # self.graph.add_edge(ticker, node_id)
            # self.embeddings[node_id] = self.model.encode(
            #     self._get_node_text(self.graph.nodes[node_id])
            # )

            # Add comprehensive analysis
            # Add growth analysis node
            growth_node_id = f"{ticker}_growth_analysis"
            self.graph.add_node(
                growth_node_id, type="growth_analysis", data=analysis.get('growth_analysis','growth')
            )
            self.graph.add_edge(ticker, growth_node_id)
            self.embeddings[growth_node_id] = self.model.encode(
                self._get_node_text(self.graph.nodes[growth_node_id])
            )

            # Add financial health node
            health_node_id = f"{ticker}_financial_health"
            self.graph.add_node(
                health_node_id, type="financial_health", data=analysis["financial_health"]
            )
            self.graph.add_edge(ticker, health_node_id)
            self.embeddings[health_node_id] = self.model.encode(
                self._get_node_text(self.graph.nodes[health_node_id])
            )

            # Add market sentiment node
            sentiment_node_id = f"{ticker}_market_sentiment"
            self.graph.add_node(
                sentiment_node_id,
                type="market_sentiment",
                data=analysis["market_sentiment"],
            )
            self.graph.add_edge(ticker, sentiment_node_id)
            self.embeddings[sentiment_node_id] = self.model.encode(
                self._get_node_text(self.graph.nodes[sentiment_node_id])
            )

            # Add summary node
            summary_node_id = f"{ticker}_analysis_summary"
            self.graph.add_node(
                summary_node_id, type="analysis_summary", data=analysis["summary"]
            )
            self.graph.add_edge(ticker, summary_node_id)
            self.graph.add_edge(summary_node_id, growth_node_id)
            self.graph.add_edge(summary_node_id, health_node_id)
            self.graph.add_edge(summary_node_id, sentiment_node_id)
            self.embeddings[summary_node_id] = self.model.encode(
                self._get_node_text(self.graph.nodes[summary_node_id])
            )

            self.logger.info(f"Completed adding data for {ticker}")
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise

    def convert_timestamps(self, data):
        """Recursively convert Timestamp objects to ISO format strings"""
        if isinstance(data, dict):
            return {k: self.convert_timestamps(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_timestamps(v) for v in data]
        elif isinstance(data, (pd.Timestamp, datetime)):
            return data.isoformat()
        else:
            return data

    @lru_cache(maxsize=1000)
    def get_total_assets(self, ticker: str) -> float:
        self.logger.debug(f"Getting total assets for: {ticker}")
        financials_node_id = f"financials_{ticker}"
        if self.graph.has_node(financials_node_id):
            financials = self.graph.nodes[financials_node_id]
            return financials.get("totalAssets", 0)
        self.logger.warning(f"No financial data found for {ticker}")
        return 0

    @lru_cache(maxsize=1000)
    def get_total_equity(self, ticker: str) -> float:
        self.logger.debug(f"Getting total equity for: {ticker}")
        financials_node_id = f"financials_{ticker}"
        if self.graph.has_node(financials_node_id):
            financials = self.graph.nodes[financials_node_id]
            return financials.get("totalStockholderEquity", 0)
        self.logger.warning(f"No financial data found for {ticker}")
        return 0

    def add_advanced_financial_metrics(self, ticker: str, financials: Dict):
        self.logger.debug(f"Adding advanced financial metrics for: {ticker}")
        # Calculate advanced metrics
        revenue = financials["revenue"]
        net_income = financials["netIncome"]
        print(financials)
        total_assets = self.get_total_assets(ticker)

        total_equity = self.get_total_equity(ticker)
        print("Total_ASSESTS & EQYIY", total_equity, total_assets)

        roa = net_income / total_assets if total_assets else None
        roe = net_income / total_equity if total_equity else None
        net_profit_margin = net_income / revenue if revenue > 0 else 0
        asset_turnover = revenue / total_assets if total_assets else None

        # DuPont Analysis
        dupont_analysis = {
            "ROE": roe,
            "Net Profit Margin": net_profit_margin,
            "Asset Turnover": asset_turnover,
            "Equity Multiplier": total_assets / total_equity if total_equity else None,
        }

        # Add new nodes
        metrics_node_id = f"{ticker}_advanced_metrics"
        self.graph.add_node(
            metrics_node_id,
            type="advanced_metrics",
            roa=roa,
            roe=roe,
            net_profit_margin=net_profit_margin,
            asset_turnover=asset_turnover,
            dupont_analysis=dupont_analysis,
        )
        self.graph.add_edge(ticker, metrics_node_id)

        # Generate insights
        insights = self.generate_financial_insights(
            ticker,
            financials,
            roa,
            roe,
            net_profit_margin,
            asset_turnover,
            dupont_analysis,
        )
        insights_node_id = f"{ticker}_financial_insights"
        self.graph.add_node(
            insights_node_id, type="financial_insights", insights=insights
        )
        self.graph.add_edge(ticker, insights_node_id)
        self.graph.add_edge(metrics_node_id, insights_node_id)

        # Update embeddings
        self.embeddings[metrics_node_id] = self.model.encode(
            self._get_node_text(self.graph.nodes[metrics_node_id])
        )
        self.embeddings[insights_node_id] = self.model.encode(
            self._get_node_text(self.graph.nodes[insights_node_id])
        )

        self.logger.info(f"Advanced metrics added for {ticker}")

    def generate_financial_insights(
        self,
        ticker: str,
        financials: Dict,
        roa: float,
        roe: float,
        net_profit_margin: float,
        asset_turnover: float,
        dupont_analysis: Dict,
    ) -> str:
        self.logger.debug(f"Generating financial insights for: {ticker}")
        insights = f"Deep Financial Insights for {ticker}:\n\n"

        # Profitability Insight
        insights += f"1. Profit Efficiency: {ticker}'s net profit margin of {net_profit_margin:.2%} indicates that for every dollar of revenue, the company retains ${net_profit_margin:.2f} as profit. This is a testament to Apple's pricing power and operational efficiency.\n\n"

        # ROE Decomposition Insight
        if roe is not None:
            insights += f"2. Return on Equity (ROE) Decomposition: The DuPont analysis reveals that {ticker}'s ROE of {roe:.2%} is driven by:\n"
        else:
            insights += f"2. Return on Equity (ROE) Decomposition: ROE data is not available for {ticker}.\n"
        insights += f"   - Profit Margin: {dupont_analysis['Net Profit Margin']:.2%}\n"
        if dupont_analysis["Asset Turnover"]:
            insights += f"   - Asset Turnover: {dupont_analysis['Asset Turnover']}\n"
        insights += f"   - Equity Multiplier: {dupont_analysis['Equity Multiplier']}\n"
        insights += "   This breakdown shows that Apple's high ROE is primarily driven by its strong profit margins rather than aggressive leverage or asset utilization.\n\n"

        # R&D Efficiency
        r_and_d_to_revenue = (
            financials["researchAndDevelopmentExpenses"] / financials["revenue"]
        )
        insights += f"3. R&D Efficiency: {ticker} invests {r_and_d_to_revenue} of its revenue in R&D. This level of investment is crucial for maintaining its innovative edge in the tech industry while still maintaining high profit margins.\n\n"

        # Cash Flow Insight (assuming you have this data)
        # operating_cash_flow = self.get_operating_cash_flow(ticker)  # You'll need to implement this method
        # cash_flow_to_income_ratio = operating_cash_flow / financials['netIncome'] if financials['netIncome'] else None
        # insights += f"4. Cash Flow Quality: The ratio of operating cash flow to net income is {cash_flow_to_income_ratio:.2f}, indicating {'high' if cash_flow_to_income_ratio > 1 else 'potential'} quality of earnings. {'This suggests strong cash generation from core operations.' if cash_flow_to_income_ratio > 1 else 'This might warrant further investigation into the company's accruals and earnings quality.'}\n\n"

        # Market Position Insight
        insights += f"5. Market Dominance: With a revenue of ${financials['revenue'] / 1e9:.2f} billion, {ticker} demonstrates its commanding position in the tech industry. This scale provides significant advantages in negotiating power, brand recognition, and economies of scale.\n\n"

        return insights

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        self.logger.debug(f"Retrieving top {k} nodes for query: {query}")
        query_embedding = self.model.encode(query)

        def calculate_similarity(item):
            node, embedding = item
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            return (node, similarity)

        similarities = [calculate_similarity(item) for item in self.embeddings.items()]

        self.logger.info(f"Retrieved {len(similarities)} nodes")
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    def _get_node_text(self, node_data: Dict) -> str:
        self.logger.debug(f"Getting node text for type: {node_data['type']}")
        if node_data["type"] == "company":
            return f"{node_data['name']} ({node_data['ticker']})"
        elif node_data["type"] == "news":
            return f"{node_data['title']} - {node_data['description']}"
        elif node_data["type"] == "transcript":
            return node_data["title"]
        elif node_data["type"] in ["financials", "profile"]:
            return json.dumps(node_data)
        elif node_data["type"] == "industry_report":
            return f"{node_data['title']} - {node_data['content']}"
        elif node_data["type"] == "yfinance":
            return (
                f"YFinance data for {node_data['info'].get('longName', '')}: "
                f"Price: {node_data['info'].get('currentPrice', 'N/A')}, "
                f"Market Cap: {node_data['info'].get('marketCap', 'N/A')}, "
                f"P/E Ratio: {node_data['info'].get('trailingPE', 'N/A')}, "
                f"Dividend Yield: {node_data['info'].get('dividendYield', 'N/A')}, "
                f"52 Week High: {node_data['info'].get('fiftyTwoWeekHigh', 'N/A')}, "
                f"52 Week Low: {node_data['info'].get('fiftyTwoWeekLow', 'N/A')}"
            )
        elif node_data["type"] == "growth_analysis":
            return f"Growth Analysis for {node_data['data']['analysis']}"
        elif node_data["type"] == "financial_health":
            return f"Financial Health Analysis: {node_data['data']['analysis']}"
        elif node_data["type"] == "market_sentiment":
            return f"Market Sentiment Analysis: {node_data['data']['analysis']}"
        elif node_data["type"] == "analysis_summary":
            return f"Analysis Summary: {node_data['data']}"
        elif node_data["type"] == "advanced_metrics":
            node_id = node_data.get("ticker", "")
            print(node_id, "node is")
            return (
                f"Advanced Financial Metrics for {node_id}:\n"
                f"ROA: {node_data['roa']}\n"
                f"ROE: {node_data['roe']}\n"
                f"Net Profit Margin: {node_data['net_profit_margin']}\n"
                f"Asset Turnover: {node_data['asset_turnover']}\n"
                f"DuPont Analysis: {node_data['dupont_analysis']}"
            )
        elif node_data["type"] == "financial_insights":
            return node_data["insights"]
        else:
            return str(node_data)

    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        relevant_nodes = self.retrieve(question)
        context = "\n".join(
            [self._get_node_text(self.graph.nodes[node]) for node, _ in relevant_nodes]
        )
        node_ids = [node for node, _ in relevant_nodes]
        return context, node_ids


class Chat:
    def __init__(self, graph_rag):
        self.openai_key =config.OPENAI_API_KEY
        self.elevenlabs_key = config.ELEVENLABS_API_SECRET_KEY

        self.openai_client = AsyncOpenAI(api_key=self.openai_key)
        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_key)
        self.system_prompt = {"role": "system", "content": system_prompt}

        self.model = faster_whisper.WhisperModel(
            model_size_or_path="tiny.en",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.history = []
        self.graph_rag = graph_rag

    async def generate_audio_stream(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        assistant_reply = ""

        # Retrieve context and node IDs from GraphRAG
        context, node_ids = self.graph_rag.answer_question(user_text)

        # Send node IDs to the client immediately
        yield {"type": "node_ids", "data": node_ids}

        # Generate the assistant's reply using OpenAI's API
        messages = [
            self.system_prompt,
            {"role": "user", "content": f"Context: {context}\nUser: {user_text}"}
        ] + self.history[-10:]

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        
        

        buffer = ""
        async for chunk in response:
            if content := chunk.choices[0].delta.content:
                assistant_reply += content
                buffer += content
                yield {"type": "reply", "data":content}
                # Convert buffer to audio when it reaches a certain length or contains punctuation
                if len(buffer) > 50 or any(char in buffer for char in '.!?'):
                    audio_chunk = await self.text_to_speech_stream(buffer)
                    yield {"type": "audio", "data": audio_chunk}
                    buffer = ""
        
        context_chart = ContextBasedChartGenerator()
        insights = context_chart.generate_dynamic_chart_data(user_text,context)
        yield {"type": "chart", "data": insights}
        # Convert any remaining text in the buffer
        if buffer:
            audio_chunk = await self.text_to_speech_stream(buffer)
            yield {"type": "audio", "data": audio_chunk}

        self.history.append({"role": "assistant", "content": assistant_reply})

    async def text_to_speech_stream(self, text):
        # Use ElevenLabs API to generate audio from text
        audio_stream = self.elevenlabs_client.generate(
            text=text,
            voice="Archer",
            model="eleven_turbo_v2_5",#"eleven_multilingual_v2",
            stream=True,
        )

        audio_chunks = b""
        for chunk in audio_stream:
            audio_chunks += chunk
        return audio_chunks

 
