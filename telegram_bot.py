from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackContext, filters
from vectordb import retrieval_qa_run
from langchain.memory import ConversationBufferMemory
import characters

# Configuration for your bot and document database
TELEGRAM_TOKEN = '6180538299:AAG8PAwJOnbzWvp8eX2g251v6CNdb40Kfd4'

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Welcome to Lagrange_QA! Send /help for instructions.")

async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text("You can ask questions using by simply typing your question.")

async def ask_command(update: Update, context: CallbackContext):                                   
       try:                                                                                           
           system_message = characters.CUSTOM_INSTRUCTIONS                                            
           context_memory = ConversationBufferMemory(input_key="question", memory_key="history")      
           question = update.message.text  # Join the words into a single string                   
           # Call functions from vectordb.py to find answers                                          
           answer_tuple = retrieval_qa_run(system_message, question, context_memory)                  
           # Only take the first element of the tuple (the answer string)                             
           answer = str(answer_tuple[0])                                                              
           # Truncate the answer if it's too long                                                     
           if len(answer) > 4090:  # Reduced limit to account for "Answer: " and "..."                
               answer = answer[:4087] + "..."                                                         
           # Send the answer back to the user                                                         
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
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))

    application.run_polling()

if __name__ == "__main__":
    main()
