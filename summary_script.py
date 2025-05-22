from main import generate_daily_summary_from_csv, send_to_discord

if __name__ == "__main__":
    text = generate_daily_summary_from_csv()
    send_to_discord("🧠 **GPT Daily Market Summary**\n" + text)
