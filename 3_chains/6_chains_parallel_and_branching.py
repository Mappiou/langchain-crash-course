import openai
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM

# Set up your OpenAI API key

# Initialize the language models (you can replace 'text-davinci-003' with any other model you'd like to use)
llm = OllamaLLM(model="tinyllama")
llm2 = OllamaLLM(model="mistral")



# Step 1: Generate two reviews in parallel

# Define prompts for the two models
review_prompt_1 = PromptTemplate(
    input_variables=["product"],
    template="Write a detailed review of the product '{product}'"
)

review_prompt_2 = PromptTemplate(
    input_variables=["product"],
    template="Write a detailed review of the product '{product}'"
)

# Create chains for both reviews
llm_chain_1 = LLMChain(llm=llm, prompt=review_prompt_1)
llm_chain_2 = LLMChain(llm=llm2, prompt=review_prompt_2)

# Step 2: Run both review chains in parallel
from concurrent.futures import ThreadPoolExecutor

def generate_reviews(product):
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Run both review generation in parallel
        future_review_1 = executor.submit(llm_chain_1.run, product)
        future_review_2 = executor.submit(llm_chain_2.run, product)

        # Get the results
        review_1 = future_review_1.result()
        review_2 = future_review_2.result()

    return review_1, review_2

# Example product
product_name = "MacBook Pro M2"

# Generate reviews
review_1, review_2 = generate_reviews(product_name)

print("Review 1:\n", review_1)
print("\nReview 2:\n", review_2)

# Step 3: Analyze the reviews and determine which one is better
analysis_prompt = PromptTemplate(
    input_variables=["review_1", "review_2"],
    template="""
    You are a highly skilled analyst. You have two product reviews. Please read both reviews and decide which one is the best in terms of quality, helpfulness, and detail.
    If Review 1 is better, respond with "Review 1 is better". If Review 2 is better, respond with "Review 2 is better". If they are of equal quality, respond with "Both reviews are equally good".
    Review 1: {review_1}
    Review 2: {review_2}
    """
)

# Analyze which review is better
analysis_chain = LLMChain(llm=llm2, prompt=analysis_prompt)
analysis_result = analysis_chain.run({"review_1": review_1, "review_2": review_2})

# Print the result of the analysis
print("\nAnalysis Result:\n", analysis_result)
