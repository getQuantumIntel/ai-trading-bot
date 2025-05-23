from main import generate_market_forecast, send_to_discord
from datetime import datetime

if __name__ == "__main__":
    # 1) Generate the forecast
    forecast_text = generate_market_forecast()
    send_to_discord("🔮 **GPT Market Forecast for Tomorrow**\n" + forecast_text)

    # 2) Persist it to forecast_log.txt (will auto-create if missing)
    try:
        with open("forecast_log.txt", "a") as f:
            f.write(f"\n\n🗓️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(forecast_text + "\n")
        print("✅ forecast_log.txt written")
    except Exception as e:
        print(f"❌ Failed to write forecast_log.txt: {e}")


