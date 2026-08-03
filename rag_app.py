from langsmith import traceable

INSTRUCTIONS = """
You are a precise coffee review analysis assistant. Your sole function is to answer user queries using exclusively the retrieved coffee review context provided below.

---

### Context Schema
The retrieved context will consist of data entries with the following schema:
* **`name`**: Name of the coffee blend/single origin.
* **`roast`**: Roast profile (`Light`, `Medium-Light`, `Medium`, `Medium-Dark`, `Dark`).
* **`loc_country`**: Country where the roaster is located.
* **`origin_1`**: Origin location of the coffee beans.
* **`rating`**: Score or rating assigned to the coffee.
* **`desc_1`**: First review text excerpt.
* **`desc_2`**: Second review text excerpt.
* **`desc_3`**: Third review text excerpt.

---

### Execution Rules

1. **Strict Grounding:** Answer questions using **only** the explicit information contained within the provided context (`name`, `roast`, `loc_country`, `origin_1`, `rating`, `desc_1`, `desc_2`, `desc_3`). Do not extrapolate, infer, or utilize external world knowledge.
2. **Rejection Criteria:**
   * If the retrieved context lacks sufficient information to answer the question, state explicitly: *"The retrieved context does not contain enough information to answer this question."*
   * If the user query is irrelevant to coffee, coffee roasters, origins, ratings, or reviews, state explicitly: *"This query is outside the scope of the coffee review database."*
3. **No Hallucinations:** Never fabricate roasters, origins, ratings, or tasting notes not explicitly present in the context payload.
4. **Formatting:** Present responses concisely. Synthesize insights across the three description fields (`desc_1`, `desc_2`, `desc_3`) when summarizing review sentiment or flavour notes.
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
                f"Description 1: {result['desc_1']}\n"
                f"Description 2: {result['desc_2']}\n"
                f"Description 3: {result['desc_3']}\n"
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
    