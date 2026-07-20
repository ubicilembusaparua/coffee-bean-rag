INSTRUCTIONS = """
# Role and Objective
You are a friendly, welcoming barista. Answer the user's question about coffee using only the provided context. Make the information easy to understand and exciting for someone who loves coffee.

# Rules
1. Rely only on the provided context. Do not invent details or tasting notes.
2. If the context does not have the answer, say: "I can't find that specific detail in my coffee records, but I'd love to help you with what we do have!"
3. Present conflicting information as options for the user to try.

# What to Include
* **The Coffee's Story:** Origin, farm, process, and roast level.
* **The Taste:** Aroma, key flavor notes, and mouthfeel.
* **Brewing Advice:** Any extraction tips mentioned.

# Tone and Formatting
* **Tone:** Warm, enthusiastic, and approachable. Avoid heavy technical jargon.
* **Format:** Use clear headers, bold text for flavors, and bullet points. Start directly with a friendly opening.

# Inputs
Context: {context}
Query: {query}
""".strip()

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()

class RAGBase():
    
    def __init__(
            self,
            index,
            llm_client,
            instructions=INSTRUCTIONS,
            prompt_template=PROMPT_TEMPLATE,
            model="gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {'desc_1': 2, 'desc_2': 1.5, 'desc_3': 1.5}
        #filter_dict = {}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
        )
    
    def build_context(self, search_results):
        context_chunks = []

        for idx, result in enumerate(search_results, start=1):
            # Construct a single, highly structured semantic block for each coffee
            chunk = (
                f"--- Document {idx} ---\n"
                f"Coffee Name: {result['name']}\n"
                f"Origin: {result['origin_1']}\n"
                f"Sensory Notes (desc_1): {result['desc_1']}\n"
                f"Flavor Profile (desc_2): {result['desc_2']}\n"
                f"Extraction Advice (desc_3): {result['desc_3']}\n"
                f"Roast Level: {result['roast']}\n"
                f"Quality Rating: {result['rating']}"
            )
            context_chunks.append(chunk)

        # Combine all isolated blocks into one unified context string for the user prompt
        context_string = "\n\n".join(context_chunks)
        return context_string
    
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query,
            context=context
        )
    
    def llm(self, prompt):

        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]
        
        response = self.llm_client.responses.create(
            model = self.model,
            input = input_messages,
        )

        return response
    

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response