def get_bot_responses():
    """
    Returns a dictionary of known intents (keys) and responses (values).
    This serves as our deterministic logic base.
    """
    return {
        "hello": "Hi there! Welcome to DecodeLabs. How can I help you today?",
        "hi": "Hello! Hope you are having a great day. What can I do for you?",
        "what is your name": "I am DecodeBot, your rule-based AI assistant.",
        "help": "I can assist you with basic queries. Try asking about my name or the company!",
        "about decodelabs": "DecodeLabs is an innovative space focused on cutting-edge AI engineering.",
        "status": "All systems are operational and functioning efficiently!"
    }

def sanitize_input(user_text):
    """
    Cleans the input by removing leading/trailing whitespaces 
    and converting it to lowercase.
    """
    return user_text.strip().lower()

def main():
    # 1. Load our rule-based knowledge base
    responses = get_bot_responses()
    fallback_message = "I'm sorry, I don't understand that command. Try asking 'help' for a list of things I know."
    
    print("==================================================")
    print("  DecodeLabs Rule-Based Chatbot Initialized!     ")
    print("  Type 'exit' or 'quit' to close the program.     ")
    print("==================================================")
    
    # 2. Infinite Loop to keep the bot active
    while True:
        try:
            # Take user input
            raw_input = input("\nYou: ")
            
            # 3. Input Sanitization
            clean_input = sanitize_input(raw_input)
            
            # 4. Check for Exit Command
            if clean_input in ['exit', 'quit']:
                print("DecodeBot: Goodbye! Have a great day ahead.")
                break # Breaks the infinite loop and terminates the program
                
            # If the user enters nothing, just skip to the next iteration
            if not clean_input:
                continue
                
            # 5. Dictionary-based Lookup with Fallback
            # .get() looks up the key in constant time O(1). 
            # If not found, it automatically returns the fallback_message.
            bot_reply = responses.get(clean_input, fallback_message)
            
            print(f"DecodeBot: {bot_reply}")
            
        except (KeyboardInterrupt, EOFError):
            # Handles unexpected terminal closures cleanly
            print("\nDecodeBot: Session interrupted. Goodbye!")
            break

if __name__ == "__main__":
    main()