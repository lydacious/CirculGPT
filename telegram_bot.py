from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackContext, filters
from vectordb import retrieval_qa_run
from langchain.memory import ConversationBufferMemory
import characters
import settings
from zep_python import ZepClient, Session, Message, Memory
import uuid
import os

# Configuration for your bot and document database
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
ZEP_BASE_URL = settings.ZEP_API_URL


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Welcome to Lagrange_QA! Send /help for instructions. You can write in any language, the bot will respond accordingly.")

async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text("You can ask questions using by simply typing your question, and use /history to see your past questions.")

async def history_command(update: Update, context: CallbackContext):
    try:
        user_id = update.message.from_user.id
        session_id = str(user_id)

        async with ZepClient(ZEP_BASE_URL) as client:
            try:
                conversation_memory = await client.memory.aget_memory(session_id)
            except Exception as memory_error:
                print(f"Memory retrieval error: {memory_error}")
        
        if conversation_memory:
            history = [message.content for message in conversation_memory.messages if message.role == "user"]
            if history:
                history_text = "\n".join(history)
                await update.message.reply_text("Here is your question history:\n" + history_text)
            else:
                await update.message.reply_text("Your question history is empty.")
        else:
            await update.message.reply_text("No conversation history found for you.")
    except Exception as e:
        print(f"Exception occurred: {e}")

async def ask_command(update: Update, context: CallbackContext):
    try:
        system_message = characters.CUSTOM_INSTRUCTIONS
        user_id = update.message.from_user.id
        session_id = str(user_id)
        context_memory = ConversationBufferMemory(return_messages=True, input_key="question", memory_key="history")
        question = update.message.text
        print(f"Session id: {session_id}")
        print(f"Question: {question}")


        # Retrieve the conversation memory for the session
        async with ZepClient(ZEP_BASE_URL) as client:
            try:
                conversation_memory = await client.memory.aget_memory(session_id)
            except Exception as memory_error:
                print(f"Memory retrieval error: {memory_error}")

        # Check if conversation memory exists
        if conversation_memory:
            # Extract the conversation history as a list of messages
            history = [message.to_dict() for message in conversation_memory.messages]
        else:
            # If no previous conversation memory exists, start with an empty history
            history = []

         # Add the current question to the conversation history
        history.append(Message(role="user", content=question).to_dict())

        # Save the updated conversation history back to the context memory
        context_memory.save_context({"question": question, "history": history}, {"output": "whats up"})

        # Call functions from vectordb.py to find answers
        answer_tuple = retrieval_qa_run(system_message, question, context_memory)
        answer = str(answer_tuple[0])

        if len(answer) > 4090:
            answer = answer[:4087] + "..."

        # Store the conversation as a Memory in the Zep session
        async with ZepClient(ZEP_BASE_URL) as client:
            messages = [
                Message(role="user", content=question),
                Message(role="bot", content=answer),
            ]
            memory = Memory(messages=messages)
            await client.memory.aadd_memory(session_id, memory)

        await update.message.reply_text(answer)
    except Exception as e:
        print(f"Exception occurred: {e}")                                                  
                                                       

async def message_handler(update: Update, context: CallbackContext):

    await ask_command(update, context)

def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))

    application.run_polling()

if __name__ == "__main__":
    main()
    