import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, SYSTEM_PROMPT

class simple_generator:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(self, query:str, context:str)->str:

        system_message=SYSTEM_PROMPT.format(context=context)

        response=self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],max_tokens=100

        )

        return response.choices[0].message.content
    
    def generate_stream(self, query:str, context:str):
        system_message=SYSTEM_PROMPT.format(context=context)

        stream=self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],max_tokens=100, stream=True

        )

        for chunk in stream:
            if chunk.choices[0].delta.get("content"):
                yield chunk.choices[0].delta.content

        
        # Quick test
if __name__ == "__main__":
    generator = simple_generator()

    test_context = """
    [Source 1: leave_policy.txt | Dept: HR]
    Employees with 0-2 years of service: 18 days per year
    Employees with 2-5 years of service: 22 days per year
    Employees with 5-10 years of service: 26 days per year
    Employees with 10+ years of service: 30 days per year
    """

    test_query = "How many leave days do I get if I've worked here for 3 years?"

    print(f"Query: {test_query}\n")
    answer = generator.generate(test_query, test_context)
    print(f"Answer: {answer}")