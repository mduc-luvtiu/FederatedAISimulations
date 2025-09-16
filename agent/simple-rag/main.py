from langchain_ollama.llms import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate 
from vector import retriever

model = OllamaLLM(model="qwen3:0.6b") 

template = """
You are an expert in answering questions about a pizza restaurant 

Here are some relevant reviews: {reviews} 

Here is the question to answer: {questions}
"""

prompt = ChatPromptTemplate.from_template(template) 
chain = prompt | model 

while True:
    print("-"*50)
    question = input("\n\nAsk your question (q to quit): ")
    if question == "q": 
        break
    
    reviews = retriever.invoke(question)
    
    result = chain.invoke({"reviews": reviews, "questions": question})
    
    print(result)