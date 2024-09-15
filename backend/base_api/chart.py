import re
from typing import List, Dict, Any
from openai import OpenAI
import random
from data import DataFetcher
import inspect
import json
import re
import random
from config import config
from typing import Dict, Any,Literal
from pydantic import BaseModel, Field, RootModel

class Chart(BaseModel):
    type: Literal['bar', 'line', 'pie', 'scatter', 'area'] = Field(..., description="Chart type (e.g., bar, line, pie, scatter, area)")
    title: str = Field(..., description="Chart title")
    data_description: str = Field(..., description="Description of data needed for this chart")
    potential_methods: List[str] = Field(..., description="Potential DataFetcher methods that might be relevant")
    ticker: str = Field(..., description="Ticker symbol of the relevant company")

class ChartList(BaseModel):
    charts:List[Chart]
    
class ContextBasedChartGenerator:
    def __init__(self):
        
        self.openai_key = config.OPENAI_API_KEY
        self.openai_client =OpenAI(api_key=self.openai_key)
        self.data_fetcher = DataFetcher()
        self.fetcher_methods = self._get_fetcher_methods()

    def _get_fetcher_methods(self) -> Dict[str, callable]:
        xc = ['get_yfinance_data','analyze_financial_health','analyze_growth_trends','analyze_industry_growth','analyze_market_sentiment','calculate_industry_concentration','get_comprehensive_analysis','get_company_profile','get_financials','get_industry_reports','get_news']
        return {
            name: method for name, method in inspect.getmembers(self.data_fetcher, inspect.ismethod)
            if name in xc
        }

    def generate_dynamic_chart_data(self, question: str, context: str) -> List[Dict[str, Any]]:
        chart_suggestions = self.get_chart_suggestions(question, context)
        
        charts_data = []
        for suggestion in chart_suggestions:
            print(suggestion)
            chart_data = self.intelligent_data_extraction(suggestion, context)
            if chart_data:
                charts_data.append(chart_data)

        return charts_data[:4]

    def get_chart_suggestions(self, question: str, context: str) -> List[Dict[str, Any]]:
        prompt = f"""
        Given the following question and context, suggest up to 4 charts that would best visualize the relevant information. For each chart, provide:
        1. Chart type (e.g., bar, line, pie, scatter, area)
        2. Chart title
        3. What data should be extracted for this chart
        4. Potential DataFetcher methods that might be relevant (choose from: {', '.join(self.fetcher_methods.keys())})
        5.ticker symbol of relevant company
        Question: {question}
        Context: {context}

      
        """

        response = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",#"gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data visualization expert."},
                {"role": "user", "content": prompt}
            ],
            response_format=ChartList #{"type":"json_object"}
            
        )
        chart_list_json =response.choices[0].message.parsed.model_dump_json(indent=2)

 
        return eval(chart_list_json)['charts'] 

    def intelligent_data_extraction(self, suggestion: Dict[str, Any], context: str) -> Dict[str, Any]:
        ticker = suggestion['ticker']
        
        fetched_data = {}
        for method_name in suggestion['potential_methods']:
            if method_name in self.fetcher_methods:
                try:
                    method = self.fetcher_methods[method_name]
                    sig = inspect.signature(method)
                    args = {}
                    if 'ticker' in sig.parameters:
                        args['ticker'] = ticker
                    fetched_data[method_name] = method(**args)
                except Exception as e:
                    print(f"Error calling {method_name}: {str(e)}")

        return self.process_fetched_data(suggestion, fetched_data, context)

    def extract_ticker(self, context: str) -> str:
        tickers = re.findall(r'\b[A-Z]{1,5}\b', context)
        return tickers[0] if tickers else ""

    def process_fetched_data(self, suggestion: Dict[str, Any], fetched_data: Dict[str, Any], context: str) -> Dict[str, Any]:
        prompt = f"""
        Given the following chart suggestion, fetched data, and original context, create a chart that best represents the required information. If the fetched data is insufficient, use the context to fill in gaps or generate plausible data.
        ensure its insanely crazy insights and analytics, things that have a wow fucking factor, that will be so helpful, make it relatable,
        Chart Suggestion:
        {suggestion}

        Fetched Data:
        {fetched_data}

        Original Context:
        {context}

        Create a chart using the following JSON format:
        {{
            "title": "Chart Title",
            "type": "chart_type",
            "data": {{
                "labels": [...],
                "datasets": [
                    {{
                        "label": "Dataset Label",
                        "data": [...]
                    }},
                    ...
                ]
            }},
            "explanation": "Explanation of data sources and any data generation or estimation"
        }}
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis and visualization expert."},
                {"role": "user", "content": prompt}
            ],response_format={"type":"json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def generate_plausible_data(self, data_type: str, num_points: int) -> List[float]:
        """Generate plausible data when exact data is not available."""
        if data_type == "percentage":
            return [random.uniform(0, 100) for _ in range(num_points)]
        elif data_type == "large_number":
            return [random.uniform(1e6, 1e9) for _ in range(num_points)]
        elif data_type == "small_number":
            return [random.uniform(0, 100) for _ in range(num_points)]
        else:
            return [random.uniform(0, 1000) for _ in range(num_points)]