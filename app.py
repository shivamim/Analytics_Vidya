import requests
from bs4 import BeautifulSoup
import pandas as pd
import gradio as gr
import torch
from transformers import BertTokenizer, BertModel, pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json
import logging
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

class CourseSearchEngine:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.df = pd.DataFrame()
        self.last_update = None
        self.user_history = []  # Store user's history of course searches or selections
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('course_search.log'), logging.StreamHandler()]
        )

    async def fetch_courses(self) -> None:
        """Asynchronously fetch and process courses with error handling."""
        try:
            url = "https://courses.analyticsvidhya.com/pages/all-free-courses"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            courses = []
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                course_cards = soup.find_all('header', class_='course-card__img-container')
                futures = [
                    executor.submit(self._process_course_card, card)
                    for card in course_cards
                ]
                courses = [f.result() for f in futures if f.result()]

            self.df = pd.DataFrame(courses)
            self._generate_embeddings()
            self._extract_course_metadata()
            self.last_update = datetime.now()
            
            # Cache the results
            self._cache_courses()
            
        except Exception as e:
            logging.error(f"Error fetching courses: {str(e)}")
            self._load_cached_courses()

    def _process_course_card(self, card: BeautifulSoup) -> Dict[str, str]:
        """Process individual course cards with enhanced metadata extraction."""
        try:
            img_tag = card.find('img', class_='course-card__img')
            if not img_tag:
                return None
                
            title = img_tag.get('alt', '')
            image_url = img_tag.get('src', '')
            
            link_tag = card.find_previous('a')
            course_link = link_tag.get('href', '') if link_tag else ''
            if course_link and not course_link.startswith('http'):
                course_link = 'https://courses.analyticsvidhya.com' + course_link
            
            # Extract additional metadata
            description = self._extract_description(card)
            topics = self._extract_topics(title)
            difficulty = self._estimate_difficulty(title, description)
            
            return {
                'title': title,
                'image_url': image_url,
                'course_link': course_link,
                'description': description,
                'topics': topics,
                'difficulty': difficulty,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error processing course card: {str(e)}")
            return None

    def _extract_topics(self, title: str) -> List[str]:
        """Extract relevant topics from course title using NLP."""
        common_topics = ['python', 'machine learning', 'data science', 'deep learning', 
                        'analytics', 'visualization', 'statistics', 'ai']
        return [topic for topic in common_topics if topic.lower() in title.lower()]

    def _estimate_difficulty(self, title: str, description: str) -> str:
        """Estimate course difficulty based on title and description."""
        text = f"{title} {description}".lower()
        beginner_keywords = ['beginner', 'basic', 'introduction', 'fundamental']
        advanced_keywords = ['advanced', 'expert', 'complex', 'professional']
        
        beginner_count = sum(1 for keyword in beginner_keywords if keyword in text)
        advanced_count = sum(1 for keyword in advanced_keywords if keyword in text)
        
        if advanced_count > beginner_count:
            return 'Advanced'
        elif beginner_count > advanced_count:
            return 'Beginner'
        return 'Intermediate'

    def _generate_embeddings(self) -> None:
        """Generate BERT embeddings for course titles with batching."""
        self.df['embedding'] = self.df['title'].apply(
            lambda x: self._get_bert_embedding(x)
        )

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding with better tokenization handling."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def search_courses(self, query: str, filters: Dict = None) -> List[Dict]:
        """Enhanced search with filtering and ranking."""
        try:
            query_embedding = self._get_bert_embedding(query)
            filtered_df = self._apply_filters(self.df, filters)
            
            # Compute similarities
            similarities = cosine_similarity(
                query_embedding, 
                np.vstack(filtered_df['embedding'].values)
            ).flatten()
            
            # Add similarity scores and rank results
            filtered_df['score'] = similarities
            filtered_df['rank'] = filtered_df['score'].rank(ascending=False)
            
            # Apply intelligent ranking boosts
            filtered_df = self._apply_ranking_boosts(filtered_df, query)
            
            # Get top results
            top_results = filtered_df.nsmallest(10, 'rank')[['title', 'image_url', 'course_link', 'score', 'difficulty', 'topics']]
            
            # Save the current search query in the user's history for future recommendations
            self.user_history.append(query)
            self._save_user_history()

            return top_results.to_dict('records')
            
        except Exception as e:
            logging.error(f"Error in search: {str(e)}")
            return []

    def _apply_ranking_boosts(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Apply intelligent ranking boosts based on various factors."""
        df = df.copy()
        
        # Boost exact matches
        df.loc[df['title'].str.contains(query, case=False, regex=False), 'rank'] *= 0.8
        
        # Boost based on difficulty appropriateness
        if any(keyword in query.lower() for keyword in ['beginner', 'start', 'basic']):
            df.loc[df['difficulty'] == 'Beginner', 'rank'] *= 0.9
        elif any(keyword in query.lower() for keyword in ['advanced', 'expert']):
            df.loc[df['difficulty'] == 'Advanced', 'rank'] *= 0.9
            
        # Boost courses with more relevant topics
        df['topic_relevance'] = df['topics'].apply(
            lambda x: len([t for t in x if t.lower() in query.lower()])
        )
        df['rank'] = df['rank'] * (1 - df['topic_relevance'] * 0.05)
        
        return df

    def _save_user_history(self):
        """Save user history to a file or database for future reference."""
        try:
            with open('user_history.json', 'w') as f:
                json.dump(self.user_history, f)
        except Exception as e:
            logging.error(f"Error saving user history: {str(e)}")

    def get_recommendations_based_on_history(self) -> List[Dict]:
        """Recommend courses based on the user's previous searches."""
        if not self.user_history:
            return []
        
        recent_query = self.user_history[-1]
        return self.search_courses(recent_query)

def create_gradio_interface() -> gr.Interface:
    """Create an enhanced Gradio interface with additional features."""
    search_engine = CourseSearchEngine()
    
    async def initialize():
        await search_engine.fetch_courses()
    
    # Run initialization
    asyncio.run(initialize())
    
    def search_with_filters(
        query: str,
        difficulty: str = "All",
        min_relevance: float = 0.0
    ) -> str:
        filters = {
            'difficulty': None if difficulty == "All" else difficulty,
            'min_relevance': min_relevance
        }
        
        results = search_engine.search_courses(query, filters)
        return create_html_output(results)

    def create_html_output(results: List[Dict]) -> str:
        if not results:
            return '<p class="no-results">No matching courses found.</p>'
            
        html_output = '<div class="results-container">'
        for item in results:
            html_output += f'''
            <div class="course-card">
                <img src="{item['image_url']}" alt="{item['title']}" class="course-image"/>
                <div class="course-info">
                    <h3>{item['title']}</h3>
                    <p class="difficulty {item['difficulty'].lower()}">{item['difficulty']}</p>
                    <p class="topics">Topics: {', '.join(item['topics'])}</p>
                    <p>Relevance: {round(item['score'] * 100, 2)}%</p>
                    <a href="{item['course_link']}" target="_blank" class="course-link">
                        View Course
                    </a>
                </div>
            </div>'''
        html_output += '</div>'
        return html_output

    # Enhanced interface with additional controls
    return gr.Interface(
        fn=search_with_filters,
        inputs=[ 
            gr.Textbox(label="Search Query", placeholder="Enter your learning interests..."),
            gr.Dropdown(choices=["All", "Beginner", "Intermediate", "Advanced"], label="Difficulty Level", value="All"),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="Minimum Relevance Score", step=0.1)
        ],
        outputs=gr.HTML(label="Search Results"),
        title="Smart Course Discovery System",
        description="Find the perfect learning resources tailored to your needs",
        theme="huggingface",
        examples=[["machine learning for beginners", "Beginner", 0.3], ["advanced deep learning techniques", "Advanced", 0.5]]
    )

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()

