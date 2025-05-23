from main import generate_market_forecast, send_to_discord
from datetime import datetime

if __name__ == "__main__":
    forecast_text = generate_market_forecast()
    send_to_discord("🔮 **GPT Market Forecast for Tomorrow**\n" + forecast_text)

    # Log each forecast for later analysis
    with open("forecast_log.txt", "a") as f:
        f.write(f"\n\n🗓️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(forecast_text + "\n")

