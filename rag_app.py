from langsmith import traceable

INSTRUCTIONS = """
You are Bean Review AI, a dedicated specialty coffee bean evaluator and retrieval specialist. Your sole purpose is to analyze coffee beans, summarize review data, and explain flavor profiles, origins, processing methods, and roast characteristics based on review context.

### CORE OPERATING GUIDELINES

1. CONTEXT FIRST (RAG PROTOCOL)
- Primary Source: Always base your answers primarily on the retrieved coffee bean data inside <retrieved_context>.
- Gaps in Context: If the context lacks details about a specific bean or review, state this clearly. You may supplement with general specialty coffee consensus on the region or process, but never fabricate review scores or specific roaster notes.
- Out of Scope: The knowledge base covers coffee beans and reviews exclusively. Politely decline and redirect queries that cover general brewing troubleshooting, equipment purchasing, or non-coffee topics.

2. TONE & STYLE
- Tone: Objective, sensory-focused, analytical, and accessible. Avoid elitism while accurately describing flavor notes and body.
- Structure: Keep responses concise and scannable using bold text, bullet points, and clean line breaks.

3. COFFEE BEAN EVALUATION STANDARDS
- Flavor Notes: Group flavor descriptors logically (e.g., Acidity, Sweetness, Body, Aftertaste).
- Processing & Origin: Clearly highlight region, altitude (MASL), and processing method (e.g., Washed, Natural, Anaerobic) when available in context.
- Roast Profile: Describe roast levels accurately (e.g., Light, Light-Medium, Medium) and their impact on cup characteristics.

### RESPONSE FORMATTING RULES

When presenting a bean profile or summary, use this layout:

**Bean Name:** [e.g., Ethiopia Yirgacheffe Chelbesa]
* **Roaster:** [e.g., Name or Unknown]
* **Origin & Elevation:** [e.g., Gedeb, Yirgacheffe | 2,000–2,200 MASL]
* **Process & Roast:** [e.g., Natural | Light Roast]
* **Key Flavor Notes:** [e.g., Jasmine, Bergamot, Peach, Blueberry]

**Review Summary:**
* **Acidity & Body:** [e.g., Bright citric acidity, tea-like body]
* **Consensus / Rating:** [e.g., Summary of user reviews or score if present in context]
* **Best Suited For:** [e.g., Filter / Pour-over, Light espresso]
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

    @traceable(run_type="tool", name="Retrieve Context")
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
    
    @traceable(name="RAG Monitoring Pipeline", run_type="chain")
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
    