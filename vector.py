import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load IPL 2024 stats CSV file
csv_file = "ipl_complete_data_2024.csv"
df = pd.read_csv(csv_file).fillna("")

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Vector store directory
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

documents = []
ids = []

if add_documents:
    for i, row in df.iterrows():
        team1 = row.get('Team_1', '').strip()
        team2 = row.get('Team_2', '').strip()
        
        content = (
            f"Match Number: {row.get('Match_Number', '')}\n"
            f"Teams: {team1} vs {team2}\n"
            f"Matchup: {team1} versus {team2}\n"
            f"Date & Time: {row.get('Date_Time', '')}\n"
            f"Venue: {row.get('Venue', '')}\n"
            f"Team 1: {team1} (Score: {row.get('Team_1_Score', '')})\n"
            f"Team 2: {team2} (Score: {row.get('Team_2_Score', '')})\n"
            f"Winner: {row.get('winner', '')}\n"
            f"Winning Margin: {row.get('winning_margin', '')}\n"
            f"Head to head: {team1} against {team2}\n"
            f"Direct matchup between {team1} and {team2}"
        )
        
        # Normalize team names for better matching
        team1_short = team1.replace("Royal Challengers Bengaluru", "RCB").replace("Sunrisers Hyderabad", "SRH")
        team2_short = team2.replace("Royal Challengers Bengaluru", "RCB").replace("Sunrisers Hyderabad", "SRH")
        
        metadata = {
            "team1score": str(row.get("Team_1_Score", "")),
            "team2score": str(row.get("Team_2_Score", "")),
            "team1": team1,
            "team2": team2,
            "team1_short": team1_short,
            "team2_short": team2_short,
            "matchup": f"{team1} vs {team2}",
            "winner": str(row.get('winner', '')),
            "match_number": str(row.get('Match_Number', ''))
        }
        document = Document(
            page_content=content,
            metadata=metadata,
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="ipl_standings",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    # No persist() call here

retriever = vector_store.as_retriever(search_kwargs={"k": 20})

# Export vector_store for use in local.py
__all__ = ['retriever', 'vector_store']