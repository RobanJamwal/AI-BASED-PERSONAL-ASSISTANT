





!pip install SpeechRecognition

import speech_recognition as sr

# Create a Recognizer instance
recognizer = sr.Recognizer()





with sr.Microphone() as source:
 print("Speak something...")
 audio_data = recognizer.listen(source)

!pip install transformers

from transformers import Conversation, AutoModelForCausalLM, AutoTokenizer

def main():
    # Load pre-trained model and tokenizer
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Begin conversation loop
    print("Personal Assistant: Hello! How can I assist you today?")
    conversation = Conversation()
    while True:
        user_input = input("You: ")
        conversation.add_user_input(user_input)

        # Generate response
        input_ids = tokenizer.encode(str(conversation), return_tensors="pt")
        response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        print("Personal Assistant:", response)
        conversation.add_response(response)

if __name__ == "__main__":
    main()

from transformers import pipeline

# Load text generation pipeline
text_generator = pipeline("text-generation")

def personal_assistant(prompt):
    # Generate response based on the prompt
    response = text_generator(prompt, max_length=50, do_sample=True, temperature=0.7)[0]['generated_text']
    return response

# Example usage
prompt = "How are you"
print(personal_assistant(prompt))
while(True):
  prompt = input("user: ")
  print(personal_assistant(prompt))

from transformers import pipeline
pipe = pipeline(model="suno/bark-small")
output = pipe("Hey it's HuggingFace on the phone!")

audio = output["audio"]
sampling_rate = output["sampling_rate"]
