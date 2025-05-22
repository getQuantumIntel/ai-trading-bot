from main import run_bot, send_to_discord

if __name__ == "__main__":
    output = run_bot()
    send_to_discord(output)
