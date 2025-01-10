from transformers import AutoModelForCausalLM, AutoTokenizer

def chatbot():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Chatbot: Hi! Ask me anything. Type 'exit' to end the chat.")
    chat_history_ids = None

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Bye! Have a great day!")
            break

        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        chat_history_ids = new_input_ids if chat_history_ids is None else \
            torch.cat([chat_history_ids, new_input_ids], dim=-1)

        response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
