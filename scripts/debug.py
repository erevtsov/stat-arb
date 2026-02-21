from analysis.preprocessing import load_processed

# fetch_1min_chunk(
#     ticker="AAPL",
#     start="2025-01-01",
#     end="2025-01-15",
#     api_key=os.environ['EODHD_API'],
# )
# df = preprocess_ticker(ticker="AAPL")

df = load_processed(
    ticker="AAPL", timeframe="15min", start_date="2022-01-03", end_date="2023-01-03"
)
x = 5
