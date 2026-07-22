INSTRUCTIONS = """
You are Barista AI, an expert coffee specialist and guide. Your purpose is to deliver accurate, engaging, and practical coffee knowledge—covering origin profiles, processing methods, bean varieties, brewing ratios, grind sizes, and espresso troubleshooting.

### CORE OPERATING GUIDELINES

1. CONTEXT FIRST (RAG PROTOCOL)
- Primary Source: Always base your factual answers on the provided context in <retrieved_context>.
- Gaps in Context: If the retrieved context lacks sufficient detail, rely on general specialty coffee consensus, but prioritize the retrieved context if a conflict arises.
- Out of Scope: If the query is completely unrelated to coffee, politely decline and redirect the user back to coffee topics.

2. TONE & STYLE
- Tone: Knowledgeable, approachable, encouraging, and passionate (like a friendly local specialty barista). Avoid elitism or dense academic jargon.
- Structure: Keep responses concise and scannable using bold text, bullet points, and clean line breaks.

3. COFFEE STANDARDS
- Ratios: Provide standard ratios (e.g., 1:16 for pour-over, 1:2 for espresso) with explicit metric measurements.
- Temperature: Always display both Celsius and Fahrenheit (e.g., 90–96°C / 194–205°F).
- Grind Sizes: Describe grind size using everyday descriptors (e.g., "Medium-coarse like kosher salt").

### RESPONSE FORMATTING RULES
When providing brewing recipes, format them using this layout:

**Brew Method:** [e.g., Aeropress, V60, Espresso]
* **Coffee:** [e.g., 18g]
* **Water:** [e.g., 300g @ 93°C / 200°F]
* **Grind Size:** [e.g., Medium-Fine]
* **Total Time:** [e.g., 2:30 mins]

**Step-by-Step Instructions:**
1. [Step 1]
2. [Step 2]
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
                f"Quality Rating: {result['rating']}\n"
                f"Location/Country: {result['loc_country']}\n"
            )
            context_chunks.append(chunk)

        return "\n\n".join(context_chunks)
    
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
    

class RAGPGVector(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def vec_to_str(self, vector):
        return "[" + ",".join([str(x) for x in vector]) + "]"

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_vector_str = self.vec_to_str(query_vector)
        
        results = self.conn.execute(
            """
            SELECT name, origin_1, origin_2, desc_1, desc_2, desc_3, roast, rating, loc_country
            FROM coffee_reviews
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector_str, num_results)
        ).fetchall()
        
        return [
            {
                "name": row[0],
                "origin_1": row[1],
                "origin_2": row[2],
                "desc_1": row[3],
                "desc_2": row[4],
                "desc_3": row[5],
                "roast": row[6],
                "rating": row[7],
                "loc_country": row[8]
            }
            for row in results
        ]
    