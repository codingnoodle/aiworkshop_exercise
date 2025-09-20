# 🧬 Biopharma M&A Radar

A real-time web application for tracking biotech and biopharma mergers, acquisitions, and partnerships using the GDELT 2.0 DOC API.

## 🚀 Quick Start

### 1. Setup & Installation
```bash
# Navigate to project directory
cd '/Users/junchilu/Desktop/2025/AI & Data Science/00_Projects/easy_prompt_app'

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App
```bash
# Option 1: Direct command
streamlit run app.py

# Option 2: Use startup script
./start_app.sh
```

### 3. Access the App
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.0.13:8501

## 📁 Project Structure

```
easy_prompt_app/
├── app.py                    # 🎯 MAIN APPLICATION (635 lines)
├── requirements.txt          # 📦 Python dependencies
├── start_app.sh             # 🚀 Quick launcher script
├── venv/                    # 🐍 Virtual environment
└── README.md                # 📖 This comprehensive guide
```

**Single Source of Truth**: Everything is in `app.py` - no confusion about which file to run!

## 🎯 Features

### Real-time Data
- **GDELT 2.0 DOC API Integration**: Fetches biotech news from thousands of sources
- **No API Key Required**: Free public API access
- **Smart Filtering**: Automatically filters for biotech/pharma deals

### Intelligent Data Extraction
- **Deal Size Parsing**: Converts "$2.3B", "800 million", "1.2bn" to USD millions
- **Company Extraction**: Identifies acquirer and target companies
- **Deal Type Classification**: Acquisition, merger, partnership, investment
- **Deduplication**: Removes duplicate articles by URL

### Interactive Visualizations
- **KPI Dashboard**: Total deals, deals with size, median/largest deal values
- **Bar Chart**: Top 10 deals by size (Acquirer → Target)
- **Timeline**: Deal dots over time with size proportional to value
- **Network Graph**: Acquirer-target relationships with edge thickness = deal size
- **Data Table**: Complete deal information with clickable source links

## 🛠️ Tech Stack

- **Backend**: Python 3.13.5
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, NetworkX, Matplotlib, Seaborn
- **API**: GDELT 2.0 DOC API

## 📊 How to Use

### 1. Launch the App
```bash
source venv/bin/activate
streamlit run app.py
```

### 2. Configure Search
- Use the sidebar to set search parameters
- Adjust date range (30-365 days)
- Customize search queries

### 3. Fetch Data
- Click "Fetch Deals Data" to search for recent deals
- The app automatically searches multiple biotech-related queries
- Results are filtered for biotech/pharma relevance

### 4. Explore Results
- View KPI cards with key metrics
- Analyze top deals in the bar chart
- Track deal timeline
- Explore company relationships in network graph
- Browse detailed deals table

## 🔍 Data Sources & Processing

### GDELT API Queries
The app automatically searches for:
- "biotech acquisition AND (merck OR abbvie OR pfizer OR novartis OR roche)"
- "biopharma merger AND (therapeutics OR pharmaceuticals OR biotech)"
- "pharmaceutical acquisition AND (drug OR medicine OR therapy)"
- "biotech partnership AND (drug development OR clinical trial)"
- "pharma merger AND (biotech OR therapeutics)"
- "biotech buyout AND (drug OR medicine)"

### Data Extraction
- **Deal Size Patterns**: Recognizes $2.3B, 800 million, 1.2bn USD, etc.
- **Company Names**: Extracts acquirer and target from headlines
- **Deal Types**: Classifies as acquisition, merger, partnership, investment
- **Biotech Filtering**: Keeps only biotech/pharma related deals

## 📈 Sample Results

The app successfully extracts deals like:
- **Merck nears $10 billion deal for biotech Verona** → $10,000M, Merck → Verona
- **AbbVie Finalizes Acquisition Of Capstan Therapeutics** → AbbVie → Capstan Therapeutics
- **Recursion Pharmaceuticals acquiring REV102 rights** → Recursion Pharmaceuticals → REV102

## 🔧 Configuration

### Search Parameters
- **Date Range**: 30-365 days lookback
- **Max Records**: 250 per query (configurable)
- **Search Terms**: Customizable biotech queries

### Deal Size Recognition
- `$2.3B`, `$2.3 billion`, `2.3B dollars`
- `$800M`, `$800 million`, `800M USD`
- `$500K`, `$500 thousand`, `500K dollars`

## 🚧 Current Limitations

- Basic company name extraction (could use NER)
- Limited deal type classification
- No historical data persistence
- Basic deduplication logic

## 🔮 Future Enhancements

- Named Entity Recognition (NER) for better company extraction
- Machine learning for deal type classification
- Database integration for historical tracking
- Advanced filtering and search capabilities
- Email alerts for new deals
- Export to Excel/CSV functionality

## 🆘 Troubleshooting

### If the app doesn't start:
```bash
# Check virtual environment
source venv/bin/activate
python --version

# Reinstall dependencies
pip install -r requirements.txt
```

### If you get port conflicts:
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### If API calls fail:
- Check your internet connection
- Verify GDELT API is accessible
- Check the Streamlit logs for error messages

## 📝 API Information

### GDELT 2.0 DOC API
- **Endpoint**: `https://api.gdeltproject.org/api/v2/doc/doc`
- **Authentication**: None required
- **Rate Limits**: Reasonable usage limits apply
- **Documentation**: [GDELT Project](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)

## 🤝 Contributing

This is a workshop project designed for easy modification and extension. Key areas for improvement:

1. **Data Extraction**: Enhance company name and deal size parsing
2. **Visualizations**: Add more chart types and interactive features
3. **Data Storage**: Implement database persistence
4. **UI/UX**: Improve the Streamlit interface
5. **Performance**: Optimize API calls and data processing

## 📄 License

This project is created for educational and workshop purposes.

---

## 🎨 Vibe Coding Prompt

Here's the effective prompt to create this Biopharma M&A Radar app:

```
🚀 Cursor Kick-off Prompt

I want to build a Biopharma M&A Radar web app.

Requirements:

Stack: Use Python + Streamlit (preferred for fast prototyping).

API source: Use the GDELT 2.0 DOC API (JSON, no API key required) to fetch recent news headlines about biotech / biopharma mergers, acquisitions, and partnerships.

Data extraction: For each article, parse the headline, URL, date, acquirer, target, deal size (USD), and deal type. Use regex and heuristics to capture deal size (e.g., $2.3B, 800 million, etc.) and normalize to USD numeric values.

Deduplicate by URL or similar titles. Filter only healthcare/biopharma deals.

Visualization:

KPI cards (total deals, # with deal size, median deal size, largest deal).

Bar chart: Top 10 deals by size (Acquirer → Target, Year).

Timeline: Deal dots over time, size = deal value.

Breakdown by big pharma companies: Pie chart and bar chart showing deal distribution.

Table: Show extracted fields in a clean table with links to the source articles.

Keep code clean, modular, and well-commented so I can edit easily in the workshop.

Deliverables in this session:

Create a Streamlit app skeleton with placeholders for data fetching, parsing, and visualizations.

Implement the GDELT API fetch function with a test query for "biotech acquisition" in the last 6 months.

Return raw JSON results in a Streamlit preview to confirm API works.

Key considerations:
- Handle timezone issues in date comparisons
- Improve company name extraction with known biotech/pharma company lists
- Add realistic sample deal sizes for demonstration
- Filter for recent deals only (within specified date range)
- Remove network graph visualization (too complex)
- Focus on big pharma breakdown instead
- Clean up folder structure - consolidate documentation
- Ensure single source of truth (one main app.py file)
```

---

## 🎉 Ready to Go!

**Just run: `streamlit run app.py`**

That's it! No confusion, no multiple files to choose from. Everything you need is in the main `app.py` file.

**Built with ❤️ for the biopharma community**