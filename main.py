import csv
from datetime import datetime
from flask import Flask, request
import openai
import os
import requests
import yfinance as yf
import pytz
import feedparser
from openai import OpenAI
import ta
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Bot logic


# Create the client using your API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_headlines(ticker):
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    headlines = []
    for entry in feed.entries[:3]:  # Limit to top 3
        title = entry.title
        headlines.append(f"- {title}")

    if not headlines:
        return "No recent headlines found."

    return "\n".join(headlines)


def analyze_sentiment_with_gpt(ticker, headlines):
    if not headlines:
        return "No headlines to analyze."

    prompt = f"""
You are a financial assistant. Analyze the following news headlines related to {ticker} stock.
For each headline, give a sentiment score: Positive, Neutral, or Negative.
Also explain the reason briefly.

Headlines:
{headlines}

Return in this format:
- "Headline text" → ✅ Positive (Reason)
- "Headline text" → ⚠️ Neutral (Reason)
- "Headline text" → ❌ Negative (Reason)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # You can use "gpt-3.5-turbo" if needed
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT error: {str(e)}"

def log_signals_to_csv(ticker, sector, price, percent_change, score, sentiment_summary, rsi, breakout_signal):
    filename = "signal_log.csv"
    headers = [
        "timestamp","ticker","sector","price",
        "percent_change","score","sentiment","rsi","breakout"
    ]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        now, ticker, sector, price,
        percent_change, score, sentiment_summary, rsi, breakout_signal
    ]
    # write headers if file doesn’t exist
    try:
        with open(filename, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    except FileExistsError:
        pass
    # append the row
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
def generate_daily_summary_from_csv():
    import pandas as pd

    filename = "signal_log.csv"
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return f"Error reading log file: {str(e)}"

    # Filter today's rows
    today_str = datetime.now().strftime("%Y-%m-%d")
    df_today = df[df["timestamp"].str.startswith(today_str)]
    if df_today.empty:
        print("📭 No data found for today.")
        return "No signals logged yet today."

    print(f"✅ Found {len(df_today)} signals for today.")

    # Build the input block for GPT
    summary_input = ""
    for _, row in df_today.iterrows():
        summary_input += (
            f"{row['ticker']} ({row['sector']}) | "
            f"Price: ${row['price']} | Δ{row['percent_change']}% | "
            f"Score: {row['score']}/100 | "
            f"Sentiment: {row['sentiment']} | "
            f"RSI: {row['rsi']} | "
            f"Breakout: {row['breakout']}\n"
        )

    print("📝 Sending to GPT:\n", summary_input)

    prompt = f"""
You are a financial assistant reviewing today's AI trading bot activity.

Given the following per‐stock summaries from today's signal log, write a concise market summary:
- Focus on standout performers
- Highlight patterns or surprises
- Maintain a professional, clear tone

Here’s today’s data:
{summary_input}

Return a markdown‐style summary suitable for Discord.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT Error: {e}")
        return f"GPT summary error: {str(e)}"
        
def generate_trend_analysis_from_logs(days=5):
    try:
        with open("daily_summaries.txt", "r") as f:
            logs = f.read().strip()
    except FileNotFoundError:
        return "No summary log found."

    # Split logs into blocks by day
    entries = logs.split("\n\n🗓️ ")
    entries = ["🗓️ " + e.strip() for e in entries if e.strip()]
    recent_entries = entries[-days:]

    if not recent_entries:
        return "Not enough entries to analyze trends."

    summaries_to_analyze = "\n\n".join(recent_entries)

    prompt = f"""
You are an intelligent trading assistant. Below are {len(recent_entries)} days of daily market summaries from an AI trading bot.

Please analyze the trend across these days and answer:
- Which stocks are showing consistent bullish/bearish sentiment?
- Are any stocks flipping from bullish to bearish or vice versa?
- What is the overall market tone?
- Anything worth watching next?

Summaries:
{summaries_to_analyze}

Return the analysis in a markdown-style format suitable for Discord.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT trend error: {str(e)}"

def run_bot():
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz).strftime("%Y-%m-%d %I:%M %p")

    watchlist = {
        # 🧠 AI Stocks
        "AI": {"sector": "AI"},       # C3.ai
        "PATH": {"sector": "AI"},     # UiPath
        "BBAI": {"sector": "AI"},     # BigBear.ai
        "SOUN": {"sector": "AI"},     # SoundHound AI
        "INOD": {"sector": "AI"},     # Innodata
        "AMBA": {"sector": "AI"},     # Ambarella
        "SYM": {"sector": "AI"},      # Symbotic

        # ⚛️ Quantum Computing Stocks
        "RGTI": {"sector": "Quantum"},  # Rigetti
        "IONQ": {"sector": "Quantum"},  # IonQ
        "QUBT": {"sector": "Quantum"},  # Quantum Computing Inc.
        "QBTS": {"sector": "Quantum"},  # D-Wave Quantum
        "HON": {"sector": "Quantum"},   # Honeywell (Quantinuum)

        # 🧪 Placeholder (not public yet, skip if running live)
        # "XTALPI": {"sector": "Quantum"},  # XtalPi — private
    }

    report_lines = [f"🚨 **AI Trading Bot Report — {now}**\n"]
    
    all_scores = []  # to collect ticker, score, and full output line

    for ticker, rules in watchlist.items():
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo", interval="1d")
            sector = rules.get("sector", "Unknown")

            # Ensure clean data
            df.dropna(inplace=True)

            # Get latest price
            latest_price = round(df['Close'].iloc[-1], 2)
            previous_close = round(df['Close'].iloc[-2], 2)
            percent_change = round(((latest_price - previous_close) / previous_close) * 100, 2)

            # Apply TA indicators
            sma_indicator = ta.trend.SMAIndicator(df['Close'], window=20)
            df['SMA20'] = sma_indicator.sma_indicator()
            sma_indicator_50 = ta.trend.SMAIndicator(df['Close'], window=50)
            df['SMA50'] = sma_indicator_50.sma_indicator()
            df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

            sma20 = round(df['SMA20'].iloc[-1], 2)
            sma50 = round(df['SMA50'].iloc[-1], 2)
            ema9 = round(df['EMA9'].iloc[-1], 2)
            rsi = round(df['RSI'].iloc[-1], 2)
            score = 0

            # 🔹 SMA Breakout: +30 pts if breakout condition met
            if latest_price > sma20 * 1.02 and sma20 > sma50:
                score += 30

            # 🔹 EMA Bias: +20 pts if price above EMA9
            if latest_price > ema9:
                score += 20

            # 🔹 RSI: +20 pts if within strong zone (40–60)
            if 40 <= rsi <= 60:
                score += 20
            elif 30 <= rsi < 40 or 60 < rsi <= 70:
                score += 10  # still decent

            # 🔹 Sentiment: +30 pts based on tone
            # This assumes you have a `sentiment_summary` string like:
            # - "Positive" / "Neutral" / "Negative"
            # 🔹 Fetch headlines and analyze sentiment
            headlines = fetch_headlines(ticker)
            sentiment_summary = analyze_sentiment_with_gpt(ticker, headlines)
            # 🔹 Sentiment: +30 pts based on GPT tone

            sentiment_score = 0
            if sentiment_summary and "Positive" in sentiment_summary:
                sentiment_score = 30
            elif sentiment_summary and "Neutral" in sentiment_summary:
                sentiment_score = 15
            elif sentiment_summary and "Negative" in sentiment_summary:
                sentiment_score = 0
            else:
                sentiment_summary = "Unknown"
                sentiment_score = 0

            score += sentiment_score

            # Generate signal context
            line = f"**[{sector}] {ticker}** — ${latest_price} ({percent_change}% {'📈' if percent_change > 0 else '📉'})"

            if latest_price > sma20 > sma50:
                line += "\n⚡ Momentum: Strong bullish (price above SMA20 > SMA50)"
            elif latest_price < sma20 < sma50:
                line += "\n⚠️ Momentum: Bearish (price below SMA20 and SMA50)"

            if latest_price > ema9:
                line += "\n📊 Short-term strength: Price above EMA9"
            else:
                line += "\n📉 Weak short-term: Price below EMA9"

            if rsi > 70:
                line += f"\n🔥 RSI {rsi} — Overbought"
            elif rsi < 30:
                line += f"\n🧊 RSI {rsi} — Oversold"
            else:
                line += f"\n📈 RSI {rsi} — Neutral"

            # Adaptive signal based on SMA20
            if latest_price > sma20 * 1.02:
                line += f"\n🟢 **BREAKOUT BUY**: Price is 2% above SMA20 (${sma20})"
            elif latest_price < sma20 * 0.98:
                line += f"\n🔴 **BREAKDOWN SELL**: Price is 2% below SMA20 (${sma20})"
            else:
                line += f"\n🔍 Watching: Price near SMA20 (${sma20})"
                # Build a simple breakout description to log
                if latest_price > sma20 * 1.02:
                    breakout = f"2% above SMA20 ({sma20})"
                elif latest_price < sma20 * 0.98:
                    breakout = f"2% below SMA20 ({sma20})"
                else:
                    breakout = f"near SMA20 ({sma20})"

                # Persist this signal to signal_log.csv
                log_signals_to_csv(
                    ticker=ticker,
                    sector=sector,
                    price=latest_price,
                    percent_change=percent_change,
                    score=score,
                    sentiment_summary=sentiment_summary,
                    rsi=rsi,
                    breakout_signal=breakout
                )

            all_scores.append((ticker, score, line))

        except Exception as e:
            report_lines.append(f"**{ticker}** — ❌ Error: {str(e)}")

    # Sort all tickers by score (highest first)
    all_scores.sort(key=lambda x: x[1], reverse=True)

    # Add sorted results to final report
    for _, _, line in all_scores:
        report_lines.append(line)

    report_lines.append("\n*This is not financial advice.*")
    return "\n".join(report_lines)



# Function to send result to Discord webhook
def send_to_discord(message):
    import textwrap

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url or not webhook_url.startswith("http"):
        print("❌ ERROR: Invalid webhook URL.")
        return

    # Title (optional – only included in the first chunk)
    header = "📊 **AI Trading Bot Report**\n"
    footer = "\n\n*This is not financial advice.*"

    # Safe chunk size (Discord limit is 2000)
    chunk_size = 1900

    # Full message with header and footer
    full_message = header + message.strip() + footer

    # Split by line for cleaner breaks
    lines = full_message.splitlines()
    chunks = []
    current_chunk = ""

    for line in lines:
        # +1 for newline
        if len(current_chunk) + len(line) + 1 <= chunk_size:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Send each chunk with index
    for i, chunk in enumerate(chunks):
        numbered = f"**({i+1}/{len(chunks)})**\n" + chunk
        data = {"content": numbered}
        response = requests.post(webhook_url, json=data)

        if response.status_code == 204:
            print(f"✅ Sent chunk {i+1}/{len(chunks)}")
        else:
            print(f"❌ Chunk {i+1} failed. Status: {response.status_code} | Response: {response.text}")

def generate_top_pick_summary():
    filename = "signal_log.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        return "No signals found yet."

    # Filter just today’s rows
    today_str = datetime.now().strftime("%Y-%m-%d")
    df_today = df[df["timestamp"].str.startswith(today_str)]
    if df_today.empty:
        return "No signals logged for today."

    # Pick the highest‐scoring row
    top = df_today.sort_values(by="score", ascending=False).iloc[0]

    ticker   = top["ticker"]
    sector   = top.get("sector", "Unknown")
    score    = top["score"]
    sentiment= top.get("sentiment", "N/A")
    rsi      = top.get("rsi", "N/A")
    breakout = top.get("breakout", "")
    price    = top["price"]

    prompt = f"""
You are an elite trading assistant. Summarize why this stock is today’s best pick:

- Ticker: {ticker} ({sector})
- Current Price: ${price}
- Signal Score: {score}/100
- Sentiment: {sentiment}
- RSI: {rsi}
- Breakout Signal: {breakout}

Write a 2–3 sentence rationale in confident, hedge-fund-style language.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
        )
        return f"📌 **Top Trade Pick: [{sector}] {ticker}**\n" + resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT error: {e}"

# Route that requires a token
@app.route('/run')
def home():
    token = request.args.get('token')
    correct_token = os.getenv("BOT_SECRET_TOKEN")
    if token != correct_token:
        print("❌ ERROR: Invalid or missing token.")
        return "Access denied. Invalid token.", 403

    output = run_bot()
    send_to_discord(output)
    return "Bot ran successfully."
    
@app.route('/summary')
def summary():
    token = request.args.get('token')
    correct_token = os.getenv("BOT_SECRET_TOKEN")
    if token != correct_token:
        return "Access denied. Invalid token.", 403

    summary_text = generate_daily_summary_from_csv()
    print("GPT Summary Output:\n", summary_text)  # 🪵 log it
    send_to_discord("🧠 **GPT Daily Market Summary**\n" + summary_text)
    # Log summary to a file
    with open("daily_summaries.txt", "a") as f:
        f.write(f"\n\n🗓️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(summary_text + "\n")
        
    return "Summary posted to Discord!"

@app.route('/trend')
def trend():
    token = request.args.get('token')
    correct_token = os.getenv("BOT_SECRET_TOKEN")
    if token != correct_token:
        return "Access denied. Invalid token.", 403

    trend_summary = generate_trend_analysis_from_logs()

    if not trend_summary or "GPT error" in trend_summary:
        return "No trend summary generated."

    # ✅ Send to Discord
    send_to_discord("📈 **GPT Trend Report (Last 5 Days)**\n" + trend_summary)

    # ✅ Save to trend_reports.txt
    with open("trend_reports.txt", "a") as f:
        f.write(f"\n\n🗓️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(trend_summary + "\n")

    return "Trend analysis posted to Discord and saved to file!"

@app.route('/top-pick')
def top_pick():
    token = request.args.get('token')
    if token != os.getenv("BOT_SECRET_TOKEN"):
        return "Access denied. Invalid token.", 403

    summary = generate_top_pick_summary()
    send_to_discord(summary)

    # ─── Log Top Pick ────────────────────────────────────────────────────────
    with open("top_picks.txt", "a") as f:
        f.write(f"\n\n🗓️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(summary + "\n")
    # ──────────────────────────────────────────────────────────────────────────

    return "Top pick posted to Discord and saved to file!"
        
if __name__ == "__main__":
    # Only run the web server when you call `python main.py` directly
    app.run(host='0.0.0.0', port=8080)




