from main import generate_trend_analysis_from_logs, send_to_discord

if __name__ == "__main__":
    text = generate_trend_analysis_from_logs()
    send_to_discord("📈 **GPT Trend Report**\n" + text)
