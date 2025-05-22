from main import generate_top_pick_summary, send_to_discord

if __name__ == "__main__":
    text = generate_top_pick_summary()
    send_to_discord(text)
