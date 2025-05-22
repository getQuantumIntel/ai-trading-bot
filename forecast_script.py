from main import generate_market_forecast, send_to_discord

if __name__ == "__main__":
    text = generate_market_forecast()
    send_to_discord("🔮 **GPT Market Forecast**\n" + text)
