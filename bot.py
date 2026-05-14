import os
import io
import logging
from datetime import date
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_missing = [v for v in ("TELEGRAM_TOKEN", "GEMINI_KEY") if not os.getenv(v)]
if _missing:
    raise EnvironmentError(f"Missing environment variables: {', '.join(_missing)}")

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

PROMPT = (
    "Generate a realistic image of this person in a gym setting, "
    "wearing athletic clothes, maintaining their facial features and build."
)
MAX_DAILY = 3

user_usage: dict = {}


def check_and_increment(user_id: int) -> bool:
    today = date.today()
    entry = user_usage.get(user_id)
    if entry is None or entry["date"] != today:
        user_usage[user_id] = {"count": 1, "date": today}
        return True
    if entry["count"] >= MAX_DAILY:
        return False
    entry["count"] += 1
    return True


def decrement(user_id: int):
    if user_id in user_usage:
        user_usage[user_id]["count"] = max(0, user_usage[user_id]["count"] - 1)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not check_and_increment(user_id):
        await update.message.reply_text("You've used all 3 daily generations. Come back tomorrow! 🗓")
        return

    await update.message.reply_text("⏳ Generating your gym photo, please wait...")

    try:
        file = await context.bot.get_file(update.message.photo[-1].file_id)
        photo_bytes = bytes(await file.download_as_bytearray())

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[
                types.Part.from_bytes(data=photo_bytes, mime_type="image/jpeg"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            ),
        )

        if not response.candidates:
            decrement(user_id)
            await update.message.reply_text("🚫 Request blocked by safety filters. Try a different photo.")
            return

        image_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_data = part.inline_data.data
                break

        if image_data is None:
            decrement(user_id)
            await update.message.reply_text("🚫 No image generated. Your photo may have triggered safety filters.")
            return

        uses_left = MAX_DAILY - user_usage[user_id]["count"]
        label = "generation" if uses_left == 1 else "generations"
        await update.message.reply_photo(
            photo=io.BytesIO(image_data),
            caption=f"💪 Here's your gym photo! ({uses_left} {label} left today)",
        )

    except Exception as e:
        logger.error("Error for user %s: %s", user_id, e)
        decrement(user_id)
        msg = str(e).lower()
        if any(w in msg for w in ("safety", "blocked", "harm", "recitation")):
            await update.message.reply_text("🚫 Photo blocked by safety filters. Please try a different one.")
        else:
            await update.message.reply_text("❌ Something went wrong. Please try again.")


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📸 Send me a photo and I'll generate a gym version of you!")


def main():
    app = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(~filters.PHOTO, handle_other))
    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
