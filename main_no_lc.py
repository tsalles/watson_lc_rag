from dotenv import load_dotenv
import watsonx
from watsonx import GenParams

load_dotenv()


def classify_sentiment() -> None:
    prompt = """[Evaluate Customer Sentiment]

    Act as a Customer Satisfaction specialist. Your goal is to evaluate if a customer Comment has Satisfaction 1 if such
    Comment has positive sentiment, or if a customer Comment has Satisfaction 0 otherwise. You are tasked with analyzing
    customer comments and determining the sentiment expressed in each comment. Your goal is to assign a Satisfaction
    label to each comment based on its sentiment. For comments with a positive or neutral sentiment, assign
    Satisfaction 1. For comments with a negative sentiment, assign Satisfaction 0. If you are unsure, assign
    Satisfaction 1.

    [Examples] 

    Comment: I have had a few recent rentals that have taken a very very long time, with no offer of apology.
    Satisfaction: 0
    
    Comment: I had a very nice customer care experience.
    Satisfaction: 1
    
    Comment: Today was a normal day at the bank
    Satisfaction: 1
    """

    model_params = {
        GenParams.DECODING_METHOD: watsonx.DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 2,
    }
    model = watsonx.ai.llm(model_id=watsonx.FoundationModel.FLAN_UL2.value, params=model_params)
    while True:
        comment = input("Comment: ")
        prediction = model.generate_text(prompt=f"{prompt}\n\nComment: {comment}\nSatisfaction:")
        print('Satisfaction:', prediction)


def summarize_paragraph():
    prompt = """[Paragraph Summarization]
    
    The following document is a paragraph from a scientific machine learning paper. Please read the paragraph and then
    write a short summary with up to 100 words."""
    model_params = {
        GenParams.DECODING_METHOD: watsonx.DecodingMethods.SAMPLE,
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.MAX_NEW_TOKENS: 400,
        GenParams.TOP_P: 0.65,
        GenParams.TOP_K: 50,
        GenParams.TEMPERATURE: 0.2,
    }
    model = watsonx.ai.llm(model_id=watsonx.FoundationModel.FLAN_UL2.value, params=model_params)
    while True:
        paragraph = input("Paragraph: ")
        prediction = model.generate_text(prompt=f"{prompt}\n\n---\n{paragraph}\n---\n\nSummary:")
        print('Summary:', prediction)


def bob_alice_chat():
    import time
    bob_params = {
        GenParams.DECODING_METHOD: watsonx.DecodingMethods.SAMPLE,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 25,
        GenParams.TOP_P: 1.0,
        GenParams.TOP_K: 50,
        GenParams.TEMPERATURE: 1.0,
    }

    alice_params = {
        GenParams.DECODING_METHOD: watsonx.DecodingMethods.SAMPLE,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 45,
        GenParams.TOP_P: 1.0,
        GenParams.TOP_K: 50,
        GenParams.TEMPERATURE: 0.05,
    }
    bob_model = watsonx.ai.llm(model_id=watsonx.FoundationModel.FLAN_UL2.value, params=bob_params)
    alice_model = watsonx.ai.llm(model_id=watsonx.FoundationModel.FLAN_T5_XXL.value, params=alice_params)

    max_cycles = 20
    alice_q = "What is 1 + 1?"
    print(f"[Alice][Q] {alice_q}")
    for x in range(max_cycles):
        bob_response = bob_model.generate_text(prompt=alice_q)
        print(f"[Bob][A] {bob_response}")
        bob_q = "What is " + bob_response + " + " + bob_response + "?"
        print(f"[Bob][Q] {bob_q}")
        alice_response = alice_model.generate_text(prompt=bob_q)
        print(f"[Alice][A] {alice_response}")
        alice_q = "What is " + alice_response + " + " + alice_response + "?"
        print(f"[Alice][Q] {alice_q}")
        time.sleep(0.5)


if __name__ == '__main__':
    mode = 'chat'  # 'chat'  # 'classify'  # 'summarize'
    if mode == 'classify':
        classify_sentiment()
    elif mode == 'summarize':
        summarize_paragraph()
    elif mode == 'chat':
        bob_alice_chat()
    else:
        raise NotImplementedError
