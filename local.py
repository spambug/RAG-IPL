from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, vector_store  # Import both retriever and vector_store

model = OllamaLLM(model="llama3.2")

template = """
You are an expert analyst for IPL cricket based on 2024 season data.

Here are some relevant match details from IPL 2024:
{matches}

Based on this data, answer the question or provide your prediction for IPL 2025:

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def enhanced_retrieval(question, k=20):
    """
    Enhanced retrieval that ensures comprehensive team coverage
    """
    # Try to identify team names in the question
    team_keywords = {
        'rcb': 'Royal Challengers Bengaluru',
        'royal challengers bengaluru': 'Royal Challengers Bengaluru',
        'royal challengers': 'Royal Challengers Bengaluru',
        'srh': 'Sunrisers Hyderabad',
        'sunrisers hyderabad': 'Sunrisers Hyderabad',
        'sunrisers': 'Sunrisers Hyderabad',
        'csk': 'Chennai Super Kings',
        'chennai super kings': 'Chennai Super Kings',
        'mi': 'Mumbai Indians',
        'mumbai indians': 'Mumbai Indians',
        'kkr': 'Kolkata Knight Riders',
        'kolkata knight riders': 'Kolkata Knight Riders',
        'rr': 'Rajasthan Royals',
        'rajasthan royals': 'Rajasthan Royals',
        'gt': 'Gujarat Titans',
        'gujarat titans': 'Gujarat Titans',
        'pbks': 'Punjab Kings',
        'punjab kings': 'Punjab Kings',
        'dc': 'Delhi Capitals',
        'delhi capitals': 'Delhi Capitals',
        'lsg': 'Lucknow Super Giants',
        'lucknow super giants': 'Lucknow Super Giants'
    }
    
    question_lower = question.lower()
    mentioned_teams = []
    
    for keyword, full_name in team_keywords.items():
        if keyword in question_lower:
            mentioned_teams.append(full_name)
    
    # Check if this is a team performance query (wins, losses, matches, performance, etc.)
    performance_keywords = ['win', 'wins', 'won', 'match', 'matches', 'play', 'played', 'performance', 'season', 'loss', 'losses', 'lost', 'how many', 'total', 'all']
    is_performance_query = any(keyword in question_lower for keyword in performance_keywords)
    
    # If asking about a single team's performance, get ALL their matches
    if len(mentioned_teams) >= 1 and is_performance_query:
        target_team = mentioned_teams[0]
        print(f"[DEBUG] Detected team performance query for: {target_team}")
        
        # Get ALL documents and filter by team
        all_docs = vector_store.get()['documents']
        all_metadatas = vector_store.get()['metadatas']
        all_ids = vector_store.get()['ids']
        
        team_docs = []
        for i, (doc_content, metadata, doc_id) in enumerate(zip(all_docs, all_metadatas, all_ids)):
            if metadata and (
                metadata.get('team1') == target_team or 
                metadata.get('team2') == target_team
            ):
                # Reconstruct document object
                from langchain_core.documents import Document
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                team_docs.append(doc)
        
        print(f"[DEBUG] Found {len(team_docs)} total matches for {target_team}")
        return team_docs
    
    # If asking about multiple teams or direct matchups
    elif len(mentioned_teams) >= 2:
        # Look for direct matchups between these teams
        team1, team2 = mentioned_teams[0], mentioned_teams[1]
        print(f"[DEBUG] Detected matchup query: {team1} vs {team2}")
        
        # Get ALL documents and filter by both teams
        all_docs = vector_store.get()['documents']
        all_metadatas = vector_store.get()['metadatas']
        all_ids = vector_store.get()['ids']
        
        matchup_docs = []
        for i, (doc_content, metadata, doc_id) in enumerate(zip(all_docs, all_metadatas, all_ids)):
            if metadata and (
                (metadata.get('team1') == team1 and metadata.get('team2') == team2) or
                (metadata.get('team1') == team2 and metadata.get('team2') == team1)
            ):
                from langchain_core.documents import Document
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                matchup_docs.append(doc)
        
        print(f"[DEBUG] Found {len(matchup_docs)} direct matchups between {team1} and {team2}")
        
        # Also get semantic search results for additional context
        semantic_docs = retriever.invoke(question)
        
        # Combine and deduplicate
        all_docs = matchup_docs + semantic_docs
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.metadata.get('match_number', doc.page_content[:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs[:k]
    
    # Default to semantic search for general queries
    else:
        print("[DEBUG] Using semantic search for general query")
        return retriever.invoke(question)

while True:
    print("\n\n-------------------------------")
    question = input("Ask your IPL question (q to quit): ")
    if question.lower() == "q":
        break

    # Use enhanced retrieval
    docs = enhanced_retrieval(question)
    if not docs:
        print("[ERROR] No relevant data found.")
        continue

    # Debug: Show which matches were retrieved
    print(f"\n[DEBUG] Retrieved {len(docs)} matches:")
    for doc in docs:
        match_num = doc.metadata.get('match_number', 'Unknown')
        matchup = doc.metadata.get('matchup', 'Unknown matchup')
        print(f"  - Match {match_num}: {matchup}")

    # Combine retrieved documents' content for prompt context
    matches = "\n\n".join([doc.page_content for doc in docs])

    # Generate answer or prediction
    result = chain.invoke({"matches": matches, "question": question})
    print("\n[ANSWER]")
    print(result)