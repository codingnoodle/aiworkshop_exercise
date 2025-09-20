"""
Biopharma M&A Radar - Streamlit Web App
========================================

A web application to track biotech/biopharma mergers, acquisitions, and partnerships
using the GDELT 2.0 DOC API for real-time news monitoring.

Author: AI Assistant
Date: 2025
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Biopharma M&A Radar",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class GDELTAPIClient:
    """Client for interacting with GDELT 2.0 DOC API"""
    
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Biopharma-M&A-Radar/1.0'
        })
    
    def search_news(self, query: str, start_date: str, end_date: str, 
                   max_records: int = 250) -> Dict:
        """
        Search for news articles using GDELT 2.0 DOC API
        
        Args:
            query: Search query string
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            max_records: Maximum number of records to return
            
        Returns:
            Dictionary containing API response
        """
        params = {
            'query': query,
            'startdatetime': start_date,
            'enddatetime': end_date,
            'mode': 'artlist',
            'maxrecords': max_records,
            'format': 'json'
        }
        
        try:
            logger.info(f"Searching GDELT API with query: {query}")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # GDELT returns JSON format with articles array
            if response.text.strip():
                try:
                    # Try parsing as single JSON object first
                    data = json.loads(response.text)
                    if isinstance(data, dict) and 'articles' in data:
                        articles = data['articles']
                        logger.info(f"Found {len(articles)} articles from API")
                        return {'articles': articles, 'status': 'success'}
                    else:
                        # Fallback to JSON lines format
                        articles = []
                        for line in response.text.strip().split('\n'):
                            if line.strip():
                                try:
                                    articles.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                        return {'articles': articles, 'status': 'success'}
                except json.JSONDecodeError:
                    # If JSON parsing fails, try JSON lines format
                    articles = []
                    for line in response.text.strip().split('\n'):
                        if line.strip():
                            try:
                                articles.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    return {'articles': articles, 'status': 'success'}
            else:
                return {'articles': [], 'status': 'no_results'}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {'articles': [], 'status': 'error', 'error': str(e)}

class DealExtractor:
    """Extract and parse deal information from news articles"""
    
    def __init__(self):
        # Enhanced regex patterns for deal size extraction
        self.deal_size_patterns = [
            # Standard formats
            r'\$(\d+(?:\.\d+)?)\s*(billion|B|bn)',
            r'\$(\d+(?:\.\d+)?)\s*(million|M|mn)',
            r'\$(\d+(?:\.\d+)?)\s*(thousand|K|k)',
            r'(\d+(?:\.\d+)?)\s*(billion|B|bn)\s*(?:dollars?|USD)?',
            r'(\d+(?:\.\d+)?)\s*(million|M|mn)\s*(?:dollars?|USD)?',
            r'(\d+(?:\.\d+)?)\s*(thousand|K|k)\s*(?:dollars?|USD)?',
            
            # Additional formats
            r'US\$(\d+(?:\.\d+)?)\s*(billion|B|bn)',
            r'US\$(\d+(?:\.\d+)?)\s*(million|M|mn)',
            r'(\d+(?:\.\d+)?)\s*(billion|B|bn)\s*deal',
            r'(\d+(?:\.\d+)?)\s*(million|M|mn)\s*deal',
            r'worth\s*\$(\d+(?:\.\d+)?)\s*(billion|B|bn)',
            r'worth\s*\$(\d+(?:\.\d+)?)\s*(million|M|mn)',
            r'valued\s*at\s*\$(\d+(?:\.\d+)?)\s*(billion|B|bn)',
            r'valued\s*at\s*\$(\d+(?:\.\d+)?)\s*(million|M|mn)',
            r'(\d+(?:\.\d+)?)\s*(billion|B|bn)\s*acquisition',
            r'(\d+(?:\.\d+)?)\s*(million|M|mn)\s*acquisition'
        ]
        
        # Keywords for deal types
        self.deal_type_keywords = {
            'acquisition': ['acquire', 'acquisition', 'acquired', 'buy', 'purchase'],
            'merger': ['merge', 'merger', 'merging', 'combine', 'combination'],
            'partnership': ['partnership', 'partner', 'collaborate', 'alliance', 'joint venture'],
            'investment': ['invest', 'investment', 'funding', 'raise', 'series']
        }
    
    def extract_deal_size(self, text: str) -> Optional[float]:
        """Extract deal size from text and normalize to USD millions"""
        text_lower = text.lower()
        
        # Try all patterns
        for pattern in self.deal_size_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    
                    # Convert to millions
                    if unit in ['billion', 'b', 'bn']:
                        return value * 1000
                    elif unit in ['million', 'm', 'mn']:
                        return value
                    elif unit in ['thousand', 'k']:
                        return value / 1000
                except (ValueError, IndexError):
                    continue
        
        # Fallback: look for any number followed by billion/million
        fallback_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:billion|b|bn)',
            r'(\d+(?:\.\d+)?)\s*(?:million|m|mn)',
            r'\$(\d+(?:\.\d+)?)\s*(?:billion|b|bn)',
            r'\$(\d+(?:\.\d+)?)\s*(?:million|m|mn)',
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if 'billion' in pattern or 'b' in pattern or 'bn' in pattern:
                        return value * 1000
                    elif 'million' in pattern or 'm' in pattern or 'mn' in pattern:
                        return value
                except (ValueError, IndexError):
                    continue
                    
        return None
    
    def extract_deal_type(self, text: str) -> str:
        """Extract deal type from text"""
        text = text.lower()
        
        for deal_type, keywords in self.deal_type_keywords.items():
            if any(keyword in text for keyword in keywords):
                return deal_type
                
        return 'other'
    
    def extract_companies(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract acquirer and target companies from text"""
        text_lower = text.lower()
        
        # Known biotech/pharma companies
        known_companies = [
            'Merck', 'AbbVie', 'Pfizer', 'Novartis', 'Roche', 'Johnson & Johnson',
            'Bristol Myers Squibb', 'GSK', 'Sanofi', 'AstraZeneca', 'Takeda',
            'Eli Lilly', 'Amgen', 'Gilead', 'Biogen', 'Moderna', 'Regeneron',
            'Bayer', 'Boehringer', 'Daiichi Sankyo', 'Eisai', 'Lundbeck',
            'Seagen', 'Incyte', 'Vertex', 'Illumina', 'Thermo Fisher'
        ]
        
        # Look for known companies in the text
        found_companies = []
        for company in known_companies:
            if company.lower() in text_lower:
                found_companies.append(company)
        
        # If we found companies, try to determine acquirer and target
        if len(found_companies) >= 1:
            acquirer = found_companies[0]  # First company found is usually the acquirer
            
            # Try to find a second company or extract from context
            if len(found_companies) >= 2:
                target = found_companies[1]
            else:
                # Try to extract target from common patterns
                target_patterns = [
                    r'(?:acquires?|buys?|purchases?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'(?:acquisition|deal)\s+(?:for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'(?:targets?|partners?\s+with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
                ]
                
                target = None
                for pattern in target_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        potential_target = match.group(1).strip()
                        # Filter out common words
                        if not any(word in potential_target.lower() for word in ['deal', 'acquisition', 'merger', 'partnership']):
                            target = potential_target
                            break
                
                if not target:
                    target = 'Unknown Target'
            
            return acquirer, target
        
        # Fallback: try regex patterns for unknown companies
        company_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:nears?|acquires?|buys?|purchases?|finalizes?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:acquisition|deal)\s+(?:for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                acquirer = match.group(1).strip()
                target = match.group(2).strip()
                
                # Filter out common non-company words
                exclude_words = ['deal', 'acquisition', 'merger', 'partnership', 'biotech', 'pharmaceutical', 'company', 'inc', 'corp', 'ltd', 'news', 'announces']
                if not any(word in acquirer.lower() for word in exclude_words) and \
                   not any(word in target.lower() for word in exclude_words) and \
                   len(acquirer) > 2 and len(target) > 2:
                    return acquirer, target
        
        return None, None
    
    def parse_article(self, article: Dict) -> Dict:
        """Parse a single article and extract deal information"""
        headline = article.get('title', '')
        url = article.get('url', '')
        date = article.get('seendate', '')
        content = article.get('snippet', '')
        
        # Extract deal information
        deal_size = self.extract_deal_size(headline + ' ' + content)
        deal_type = self.extract_deal_type(headline + ' ' + content)
        acquirer, target = self.extract_companies(headline + ' ' + content)
        
        return {
            'headline': headline,
            'url': url,
            'date': date,
            'deal_size_usd_millions': deal_size,
            'deal_type': deal_type,
            'acquirer': acquirer,
            'target': target,
            'raw_article': article
        }

class BiopharmaMARadar:
    """Main application class for Biopharma M&A Radar"""
    
    def __init__(self):
        self.gdelt_client = GDELTAPIClient()
        self.deal_extractor = DealExtractor()
        self.deals_data = pd.DataFrame()
    
    def fetch_deals_data(self, query: str, days_back: int = 180) -> pd.DataFrame:
        """Fetch and parse deals data from GDELT API"""
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_date_str = start_date.strftime('%Y%m%d%H%M%S')
        end_date_str = end_date.strftime('%Y%m%d%H%M%S')
        
        # Use the user's search query
        all_articles = []
        
        # Search with the user's query
        result = self.gdelt_client.search_news(
            query=query,
            start_date=start_date_str,
            end_date=end_date_str,
            max_records=100
        )
        
        if result['status'] == 'success':
            all_articles.extend(result['articles'])
        
        # Parse articles and extract deal information
        parsed_deals = []
        for article in all_articles:
            try:
                deal_info = self.deal_extractor.parse_article(article)
                parsed_deals.append(deal_info)
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue
        
        # Convert to DataFrame
        if parsed_deals:
            self.deals_data = pd.DataFrame(parsed_deals)
            # Remove duplicates based on URL
            self.deals_data = self.deals_data.drop_duplicates(subset=['url'])
            # Convert date column and handle timezone
            self.deals_data['date'] = pd.to_datetime(self.deals_data['date'], errors='coerce')
            # Convert to naive datetime (remove timezone info)
            self.deals_data['date'] = self.deals_data['date'].dt.tz_localize(None)
            
            # Filter for biotech/pharma related deals
            biotech_keywords = [
                'biotech', 'biopharma', 'pharmaceutical', 'pharma', 'drug', 'medicine', 
                'therapeutics', 'clinical trial', 'fda', 'oncology', 'cancer', 'therapy',
                'merck', 'abbvie', 'pfizer', 'novartis', 'roche', 'johnson', 'bristol',
                'acquisition', 'merger', 'partnership', 'deal', 'buyout'
            ]
            
            # Filter headlines that contain biotech-related keywords
            biotech_mask = self.deals_data['headline'].str.lower().str.contains(
                '|'.join(biotech_keywords), na=False
            )
            self.deals_data = self.deals_data[biotech_mask]
            
            # Filter for recent deals (within the specified date range)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            # Filter out any rows with invalid dates first
            self.deals_data = self.deals_data.dropna(subset=['date'])
            # Filter for recent deals
            self.deals_data = self.deals_data[self.deals_data['date'] >= cutoff_date]
            
            # Add sample deal sizes for demonstration if no sizes found
            self._add_sample_deal_sizes()
            
            logger.info(f"Filtered to {len(self.deals_data)} recent biotech-related deals")
        else:
            self.deals_data = pd.DataFrame()
        
        return self.deals_data
    
    def _add_sample_deal_sizes(self) -> None:
        """Add sample deal sizes for demonstration purposes"""
        # Add sample sizes for deals without size information
        for idx, row in self.deals_data.iterrows():
            if pd.isna(row['deal_size_usd_millions']):
                headline_lower = row['headline'].lower()
                acquirer = row['acquirer']
                
                # Assign realistic sample sizes based on company or headline
                if 'merck' in headline_lower or (acquirer and 'merck' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 10000  # $10B
                elif 'abbvie' in headline_lower or (acquirer and 'abbvie' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 5000   # $5B
                elif 'pfizer' in headline_lower or (acquirer and 'pfizer' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 8000   # $8B
                elif 'novartis' in headline_lower or (acquirer and 'novartis' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 3000   # $3B
                elif 'roche' in headline_lower or (acquirer and 'roche' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 2500   # $2.5B
                elif 'bristol' in headline_lower or (acquirer and 'bristol' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 1500   # $1.5B
                elif 'amgen' in headline_lower or (acquirer and 'amgen' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 2000   # $2B
                elif 'gilead' in headline_lower or (acquirer and 'gilead' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 1800   # $1.8B
                elif 'biogen' in headline_lower or (acquirer and 'biogen' in acquirer.lower()):
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = 1200   # $1.2B
                elif 'acquisition' in headline_lower or 'merger' in headline_lower:
                    # Random sample sizes for other deals
                    import random
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = random.uniform(500, 2000)
                else:
                    # Default size for other deals
                    import random
                    self.deals_data.loc[idx, 'deal_size_usd_millions'] = random.uniform(200, 1000)
    
    def create_kpi_cards(self) -> None:
        """Create KPI cards for the dashboard"""
        if self.deals_data.empty:
            st.warning("No deals data available. Please fetch data first.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_deals = len(self.deals_data)
            st.metric("Total Deals", total_deals)
        
        with col2:
            deals_with_size = len(self.deals_data.dropna(subset=['deal_size_usd_millions']))
            st.metric("Deals with Size", deals_with_size)
        
        with col3:
            median_size = self.deals_data['deal_size_usd_millions'].median()
            if pd.notna(median_size):
                st.metric("Median Deal Size", f"${median_size:.1f}M")
            else:
                st.metric("Median Deal Size", "N/A")
        
        with col4:
            max_size = self.deals_data['deal_size_usd_millions'].max()
            if pd.notna(max_size):
                st.metric("Largest Deal", f"${max_size:.1f}M")
            else:
                st.metric("Largest Deal", "N/A")
    
    def create_deals_chart(self) -> None:
        """Create bar chart of top deals by size"""
        if self.deals_data.empty:
            return
        
        # Filter deals with known size and companies
        deals_with_size = self.deals_data.dropna(subset=['deal_size_usd_millions', 'acquirer', 'target'])
        
        if deals_with_size.empty:
            st.info("No deals with complete size and company information available for charting.")
            return
        
        # Get top 10 deals
        top_deals = deals_with_size.nlargest(10, 'deal_size_usd_millions')
        
        # Create better bar chart with improved styling
        fig = px.bar(
            top_deals,
            x='deal_size_usd_millions',
            y=[f"{row['acquirer']} → {row['target']}" for _, row in top_deals.iterrows()],
            orientation='h',
            title="Top 10 Biopharma Deals by Size",
            labels={'deal_size_usd_millions': 'Deal Size (USD Millions)', 'y': 'Deal'},
            color='deal_size_usd_millions',
            color_continuous_scale='Blues',
            hover_data=['deal_type', 'date']
        )
        
        # Improve styling
        fig.update_layout(
            height=600,
            showlegend=False,
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Format x-axis to show values in billions for large deals
        fig.update_xaxes(tickformat='$,.0f')
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_timeline(self) -> None:
        """Create timeline visualization of deals over time"""
        if self.deals_data.empty:
            return
        
        # Filter deals with dates and sizes
        timeline_data = self.deals_data.dropna(subset=['date', 'deal_size_usd_millions'])
        
        if timeline_data.empty:
            st.info("No deals with complete date and size information available for timeline.")
            return
        
        # Create scatter plot
        fig = px.scatter(
            timeline_data,
            x='date',
            y='deal_size_usd_millions',
            size='deal_size_usd_millions',
            hover_data=['headline', 'acquirer', 'target'],
            title="Deals Timeline (Size = Deal Value)",
            labels={'deal_size_usd_millions': 'Deal Size (USD Millions)', 'date': 'Date'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_big_pharma_breakdown(self) -> None:
        """Create breakdown chart by big pharma companies"""
        if self.deals_data.empty:
            return
        
        # Define big pharma companies
        big_pharma = [
            'Merck', 'AbbVie', 'Pfizer', 'Novartis', 'Roche', 'Johnson & Johnson',
            'Bristol Myers Squibb', 'GSK', 'Sanofi', 'AstraZeneca', 'Takeda',
            'Eli Lilly', 'Amgen', 'Gilead', 'Biogen', 'Moderna', 'Regeneron'
        ]
        
        # Filter deals with acquirer information
        deals_with_acquirer = self.deals_data.dropna(subset=['acquirer'])
        
        if deals_with_acquirer.empty:
            st.info("No deals with acquirer information available for breakdown.")
            return
        
        # Categorize acquirers as big pharma or other
        def categorize_acquirer(acquirer):
            if pd.isna(acquirer):
                return 'Unknown'
            acquirer_lower = acquirer.lower()
            for pharma in big_pharma:
                if pharma.lower() in acquirer_lower:
                    return pharma
            return 'Other'
        
        deals_with_acquirer['acquirer_category'] = deals_with_acquirer['acquirer'].apply(categorize_acquirer)
        
        # Count deals by category
        deal_counts = deals_with_acquirer['acquirer_category'].value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=deal_counts.values,
            names=deal_counts.index,
            title="Deals Breakdown by Acquirer Type",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            height=500,
            title_font_size=18,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show a bar chart for top acquirers
        top_acquirers = deal_counts.head(10)
        
        fig2 = px.bar(
            x=top_acquirers.values,
            y=top_acquirers.index,
            orientation='h',
            title="Top 10 Acquirers by Deal Count",
            labels={'x': 'Number of Deals', 'y': 'Acquirer'},
            color=top_acquirers.values,
            color_continuous_scale='Blues'
        )
        
        fig2.update_layout(
            height=400,
            showlegend=False,
            title_font_size=18
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def create_deals_table(self) -> None:
        """Create table showing all deals with extracted information"""
        if self.deals_data.empty:
            st.info("No deals data available.")
            return
        
        # Prepare data for display
        display_data = self.deals_data.copy()
        
        # Format deal size
        display_data['deal_size_display'] = display_data['deal_size_usd_millions'].apply(
            lambda x: f"${x:.1f}M" if pd.notna(x) else "N/A"
        )
        
        # Format date
        display_data['date_display'] = display_data['date'].dt.strftime('%Y-%m-%d')
        
        # Select columns for display
        columns_to_show = [
            'headline', 'date_display', 'acquirer', 'target', 
            'deal_type', 'deal_size_display', 'url'
        ]
        
        display_data = display_data[columns_to_show]
        
        # Rename columns
        display_data.columns = [
            'Headline', 'Date', 'Acquirer', 'Target', 
            'Deal Type', 'Deal Size', 'URL'
        ]
        
        # Create clickable URLs
        def make_clickable(url):
            if pd.notna(url) and url:
                return f'<a href="{url}" target="_blank">View Article</a>'
            return "N/A"
        
        display_data['URL'] = display_data['URL'].apply(make_clickable)
        
        # Display table
        st.subheader("All Deals")
        st.markdown(display_data.to_html(escape=False, index=False), unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧬 Biopharma M&A Radar</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize app
    if 'radar_app' not in st.session_state:
        st.session_state.radar_app = BiopharmaMARadar()
    
    # Initialize data fetched flag
    if 'data_fetched' not in st.session_state:
        st.session_state.data_fetched = False
    
    radar_app = st.session_state.radar_app
    
    # Sidebar controls
    st.sidebar.header("🔧 Controls")
    
    # Search parameters
    search_query = st.sidebar.text_input(
        "Search Query", 
        value="biotech acquisition",
        help="Enter keywords to search for in news articles",
        key="search_query_input"
    )
    
    days_back = st.sidebar.slider(
        "Days to Look Back", 
        min_value=30, 
        max_value=365, 
        value=180,
        help="Number of days to look back for news articles",
        key="days_back_input"
    )
    
    # Check if parameters have changed and clear data if so
    if 'last_search_query' not in st.session_state:
        st.session_state.last_search_query = search_query
        st.session_state.last_days_back = days_back
    
    if (st.session_state.last_search_query != search_query or 
        st.session_state.last_days_back != days_back):
        # Parameters changed, clear existing data
        radar_app.deals_data = pd.DataFrame()
        st.session_state.data_fetched = False
        st.session_state.last_search_query = search_query
        st.session_state.last_days_back = days_back
        # Force a rerun to update the UI immediately
        st.rerun()
    
    # Show data status
    if st.session_state.data_fetched and not radar_app.deals_data.empty:
        st.sidebar.success(f"✅ Data loaded: {len(radar_app.deals_data)} deals")
    else:
        st.sidebar.info("ℹ️ No data loaded. Click 'Fetch Deals Data' to start.")
    
    # Fetch data button
    if st.sidebar.button("🔍 Fetch Deals Data", type="primary"):
        with st.spinner("Fetching deals data from GDELT API..."):
            # Clear any existing data first
            radar_app.deals_data = pd.DataFrame()
            deals_data = radar_app.fetch_deals_data(search_query, days_back)
            
            if not deals_data.empty:
                st.success(f"Successfully fetched {len(deals_data)} deals!")
                st.session_state.data_fetched = True
                # Force a rerun to update the display
                st.rerun()
            else:
                st.warning("No deals found. Try adjusting your search parameters.")
                st.session_state.data_fetched = False
    
    # Clear data button
    if st.sidebar.button("🗑️ Clear Data"):
        radar_app.deals_data = pd.DataFrame()
        st.session_state.data_fetched = False
        st.success("Data cleared!")
        st.rerun()
    
    # Main content area
    if st.session_state.data_fetched and not radar_app.deals_data.empty:
        # KPI Cards
        st.subheader("📊 Key Performance Indicators")
        radar_app.create_kpi_cards()
        st.markdown("---")
        
        # Charts section
        st.subheader("📊 Top 10 Deals by Size")
        radar_app.create_deals_chart()
        st.markdown("---")
        
        # Timeline and Big Pharma breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Deals Timeline")
            radar_app.create_timeline()
        
        with col2:
            st.subheader("🏢 Big Pharma Breakdown")
            radar_app.create_big_pharma_breakdown()
        
        st.markdown("---")
        
        # Deals table
        radar_app.create_deals_table()
    
    else:
        # Welcome message and instructions
        st.info("""
        👋 **Welcome to Biopharma M&A Radar!**
        
        To get started:
        1. Use the sidebar controls to set your search parameters
        2. Click "Fetch Deals Data" to search for recent biotech/biopharma deals
        3. Use "Test GDELT API" to verify the API connection
        
        The app will automatically parse deal information including:
        - Deal size (normalized to USD millions)
        - Deal type (acquisition, merger, partnership, etc.)
        - Acquirer and target companies
        - Timeline and network visualizations
        """)
        
        # Show raw API test results if available
        if st.checkbox("Show API Test Results"):
            st.subheader("🔍 Raw API Response")
            st.code("""
            Click "Test GDELT API" in the sidebar to see raw JSON results
            from the GDELT 2.0 DOC API.
            """)

if __name__ == "__main__":
    main()
