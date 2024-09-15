import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from openai import OpenAI
from config import config
import sqlite3
from io import StringIO

class KnowledgeGraphBuilder:
    def __init__(self, db_path='stock_data.db'):
        self.graph = nx.Graph()
        self.company_data = {}
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.openai_client = OpenAI(api_key=config().OPENAI_API_KEY)

    def load_data(self):
        print("Loading data from database...")
        self.cursor.execute('SELECT * FROM stock_analysis')
        rows = self.cursor.fetchall()
        if not rows:
            print("No data found in the database. Please run the analysis first.")
            return False
        columns = [description[0] for description in self.cursor.description]
        for row in rows:
            data = dict(zip(columns, row))
            ticker = data['ticker']
            # Deserialize JSON fields
            data['financial_ratios'] = json.loads(data['financial_ratios'])
            data['technical_indicators'] = pd.read_json(StringIO(data['technical_indicators']))
            data['news_sentiment'] = json.loads(data['news_sentiment'])
            data['market_analysis'] = json.loads(data['market_analysis'])
            #data['global_news_sentiment'] = json.loads(data['global_news_sentiment'])
            self.company_data[ticker] = data
        print(f"Loaded data for {len(self.company_data)} companies.")
        return True

    def process_financial_data(self):
        print("Processing financial data...")
        financial_features = []
        tickers = []
        for ticker, data in self.company_data.items():
            ratios = data['financial_ratios']
            features = {
                key: float(ratios.get(key, np.nan)) if ratios.get(key, np.nan) not in [None, '', 'N/A'] else np.nan
                for key in ['P/E Ratio', 'Forward P/E', 'PEG Ratio', 'Price to Book',
                            'Debt to Equity', 'Return on Equity', 'Return on Assets', 'Profit Margin']
            }
            financial_features.append(features)
            tickers.append(ticker)
    
        df = pd.DataFrame(financial_features, index=tickers).apply(pd.to_numeric, errors='coerce')
        df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        
        if df.empty:
            print("Error: No valid financial data after cleaning.")
            return
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df.index)
        
        kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        pc_df['Cluster'] = clusters
        
        for ticker in self.company_data:
            if ticker in pc_df.index:
                self.company_data[ticker]['cluster'] = int(pc_df.loc[ticker, 'Cluster'])
                self.company_data[ticker]['pc1'] = float(pc_df.loc[ticker, 'PC1'])
                self.company_data[ticker]['pc2'] = float(pc_df.loc[ticker, 'PC2'])
        
        print("Financial data processed.")

    def build_nodes(self):
        print("Building nodes...")
        for ticker, data in self.company_data.items():
            self.graph.add_node(ticker, 
                                type='company',
                                label=data['company_name'],
                                group=data.get('cluster', 0),
                                pc1=data.get('pc1', 0),
                                pc2=data.get('pc2', 0),
                                financial_ratios=data['financial_ratios'],
                                market_analysis=data.get('market_analysis', {}))
                                #global_news_sentiment=data['global_news_sentiment'])
        print("Nodes built.")

    def build_links(self):
        print("Building links...")
        # Create edges based on financial similarity
        for ticker1 in self.company_data:
            for ticker2 in self.company_data:
                if ticker1 != ticker2:
                    similarity = self.calculate_financial_similarity(ticker1, ticker2)
                    if similarity > 0.8:  # Arbitrary threshold
                        self.graph.add_edge(ticker1, ticker2, type='financial_similarity', weight=similarity)
        
        # Create edges based on news sentiment correlation
        # sentiments = {ticker: data['global_news_sentiment']['average_sentiment'] for ticker, data in self.company_data.items()}
        # for ticker1, sentiment1 in sentiments.items():
        #     for ticker2, sentiment2 in sentiments.items():
        #         if ticker1 != ticker2:
        #             correlation = abs(sentiment1 - sentiment2)
        #             if correlation < 0.1:  # Arbitrary threshold
        #                 self.graph.add_edge(ticker1, ticker2, type='sentiment_correlation', weight=1-correlation)
        
        print("Links built.")

    def calculate_financial_similarity(self, ticker1, ticker2):
        ratios1 = self.company_data[ticker1]['financial_ratios']
        ratios2 = self.company_data[ticker2]['financial_ratios']
        common_ratios = set(ratios1.keys()) & set(ratios2.keys())
        if not common_ratios:
            return 0
        differences = []
        for ratio in common_ratios:
            try:
                val1 = float(ratios1[ratio])
                val2 = float(ratios2[ratio])
                differences.append(abs(val1 - val2) / max(abs(val1), abs(val2)))
            except (ValueError, ZeroDivisionError):
                pass
        return 1 - (sum(differences) / len(differences)) if differences else 0

    def infer_relationships(self):
        print("Inferring additional relationships...")
        for ticker, data in self.company_data.items():
            prompt = (
                f"Given the financial ratios and market analysis of {data['company_name']} (Ticker: {ticker}), "
                "identify potential strategic relationships or market dynamics with other companies in the dataset. "
                "Focus on aspects that would be most relevant for investors and traders."
            )
            input_data = {
                "financial_ratios": data['financial_ratios'],
                "market_analysis": data['market_analysis']
            }
            input_text = json.dumps(input_data)
            full_prompt = prompt + "\n\nInput Data:\n" + input_text

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst assistant."},
                        {"role": "user", "content": full_prompt}
                    ]
                )
                suggestions = response.choices[0].message.content.strip()
                for other_ticker in self.company_data:
                    if other_ticker != ticker and other_ticker in suggestions:
                        self.graph.add_edge(ticker, other_ticker, type='inferred_relation')
            except Exception as e:
                print(f"Error using GPT-4 for ticker {ticker}: {e}")
        print("Additional relationships inferred.")

    def save_graph(self, output_file):
        print(f"Saving knowledge graph to {output_file}...")
        graph_data = nx.node_link_data(self.graph)
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print("Knowledge graph saved.")

    def build_graph(self):
        if not self.load_data():
            return
        self.process_financial_data()
        self.build_nodes()
        self.build_links()
        self.infer_relationships()
        self.save_graph('knowledge_graph.json')

if __name__ == "__main__":
    graph_builder = KnowledgeGraphBuilder()
    graph_builder.build_graph()