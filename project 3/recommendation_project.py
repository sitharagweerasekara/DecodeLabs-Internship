import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_item_database():
    """
    Simulates an item database (e.g., courses/tutorials available at DecodeLabs)
    with descriptive feature tags.
    """
    data = {
        'Course_ID': [1, 2, 3, 4, 5, 6],
        'Title': [
            "Introduction to Python Programming",
            "Advanced Data Science and Analytics",
            "Neural Networks and Deep Learning",
            "Web Development with HTML, CSS, and JavaScript",
            "Cloud Automation and DevOps Engineering",
            "Machine Learning Optimization and Tensors"
        ],
        'Features': [
            "python code software development basics basics",
            "python data science pandas numpy analytics",
            "neural networks deep learning ai tensors deep",
            "web design frontend development html css javascript",
            "cloud automation devops aws software deployment",
            "machine learning optimization tensors math code"
        ]
    }
    return pd.DataFrame(data)

def main():
    print("==================================================")
    print("  DecodeLabs AI Phase 3: Recommendation Engine    ")
    print("==================================================")
    
    # 1. Load Item Data
    df = get_item_database()
    print("[*] Content Database Loaded with 6 Available Courses.")
    
    # 2. Get User Input (Preferences)
    print("\nAvailable tag keywords you can try: python, neural networks, web design, cloud, optimization, etc.")
    user_input = input("Enter your skills or interests (comma-separated): ")
    
    if not user_input.strip():
        print("[-] Input cannot be empty. Terminating engine.")
        return

    # 3. Vector Mapping via TF-IDF (Penalizes generic words, rewards specific ones)
    # This directly fulfills the Page 10 requirement!
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Combine the database features with the user input to map onto the exact same vocabulary space
    all_documents = list(df['Features']) + [user_input]
    tfidf_matrix = tfidf.fit_transform(all_documents)
    
    # Separate the database vectors from the user profile vector
    database_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]
    
    # 4. Process Similarity Logic (Cosine Similarity)
    similarity_scores = cosine_similarity(user_vector, database_vectors).flatten()
    
    # Add scores to our dataframe
    df['Match_Score'] = similarity_scores
    
    # 5. Output Generation (Sort by Top-N Highest Matches)
    recommended_items = df.sort_values(by='Match_Score', ascending=False)
    
    print("\n================ TOP RECOMMENDATIONS ================")
    rank = 1
    for index, row in recommended_items.iterrows():
        # Only show items that have some degree of structural match
        if row['Match_Score'] > 0:
            print(f"{rank}. {row['Title']}")
            print(f"   [Match Confidence: {row['Match_Score'] * 100:.2f}%]")
            rank += 1
            
    if rank == 1:
        print("[-] No matching items found for your specific preferences. Try different keywords!")
    print("=====================================================")

if __name__ == "__main__":
    main()