from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.properties import StringProperty, ObjectProperty, NumericProperty
from kivy.lang import Builder
from kivy.animation import Animation
from kivy.metrics import dp
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob, Word
import wikipediaapi
import json
import random
import threading
import time
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
import difflib

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv('write  your api key here')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Initialize the model
    try:
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        model = None
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables")
    model = None

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class KnowledgeBase:
    def __init__(self):
        self.categories = {
            "science": {
                "physics": ["quantum mechanics", "relativity", "gravity", "atoms", "energy", "force", "motion", "light", "sound", "electricity", "magnetism", "nuclear", "particle"],
                "chemistry": ["elements", "compounds", "reactions", "molecules", "periodic table", "acids", "bases", "organic", "inorganic", "solutions", "bonds", "metals", "gases"],
                "biology": ["cells", "genetics", "evolution", "organisms", "ecology", "anatomy", "physiology", "plants", "animals", "bacteria", "viruses", "dna", "proteins"],
                "astronomy": ["planets", "stars", "galaxies", "universe", "solar system", "black holes", "space", "moon", "sun", "comets", "asteroids", "nebula", "telescope"]
            },
            "history": {
                "ancient": ["egypt", "rome", "greece", "mesopotamia", "civilization", "pyramids", "empire", "pharaohs", "alexander", "caesar", "cleopatra"],
                "medieval": ["middle ages", "castle", "knight", "feudal", "crusades", "medieval europe", "vikings", "samurai", "mongols", "byzantine"],
                "modern": ["world war", "revolution", "industrial", "cold war", "democracy", "technology", "renaissance", "colonization", "independence"],
                "civilizations": ["maya", "inca", "aztec", "china", "india", "persia", "ottoman", "african kingdoms", "native american", "polynesian"]
            },
            "geography": {
                "physical": ["mountains", "rivers", "oceans", "continents", "climate", "weather", "landforms", "volcanoes", "earthquakes", "glaciers"],
                "human": ["population", "cities", "countries", "culture", "economy", "politics", "society", "migration", "urbanization", "development"],
                "environmental": ["ecosystem", "pollution", "conservation", "resources", "climate change", "biodiversity", "sustainability", "renewable"]
            },
            "technology": {
                "computers": ["software", "hardware", "programming", "internet", "artificial intelligence", "cybersecurity", "cloud computing", "databases"],
                "innovation": ["invention", "startup", "entrepreneur", "innovation", "technology", "future", "robotics", "automation", "biotechnology"],
                "digital": ["social media", "digital marketing", "e-commerce", "mobile", "apps", "web", "virtual reality", "blockchain", "cryptocurrency"]
            },
            "arts": {
                "visual": ["painting", "sculpture", "photography", "architecture", "design", "art history", "drawing", "digital art", "animation"],
                "performing": ["music", "dance", "theater", "film", "concert", "performance", "opera", "ballet", "acting", "directing"],
                "literature": ["books", "poetry", "novels", "authors", "writing", "literary", "fiction", "non-fiction", "drama", "essays"]
            },
            "sports": {
                "team": ["football", "basketball", "soccer", "baseball", "hockey", "volleyball", "cricket", "rugby"],
                "individual": ["tennis", "golf", "swimming", "athletics", "gymnastics", "boxing", "martial arts"],
                "olympics": ["summer games", "winter games", "medals", "athletes", "records", "competitions"]
            },
            "health": {
                "medical": ["medicine", "disease", "treatment", "surgery", "pharmacy", "diagnosis", "healthcare"],
                "wellness": ["nutrition", "exercise", "mental health", "meditation", "yoga", "fitness", "diet"],
                "research": ["clinical trials", "medical research", "breakthroughs", "vaccines", "therapy"]
            },
            "culture": {
                "traditions": ["customs", "festivals", "ceremonies", "rituals", "celebrations", "holidays"],
                "food": ["cuisine", "cooking", "recipes", "ingredients", "restaurants", "beverages"],
                "lifestyle": ["fashion", "trends", "entertainment", "media", "social norms", "etiquette"]
            }
        }
        
        self.knowledge_patterns = {
            "definition": r"what (is|are) (a |an |the )?([a-zA-Z\s]+)",
            "explanation": r"(explain|tell me about|describe|elaborate on) ([a-zA-Z\s]+)",
            "how_works": r"how (does|do|can|could|would|will) ([a-zA-Z\s]+) (work|function|operate)",
            "history": r"(when|what|who|where) (was|were|did|has|have) ([a-zA-Z\s]+)",
            "comparison": r"(compare|difference between|vs|versus|or|better) ([a-zA-Z\s]+)",
            "location": r"where (is|are|can i find|do we see) ([a-zA-Z\s]+)",
            "time": r"when (did|does|will|should|can|would) ([a-zA-Z\s]+)",
            "why": r"why (is|are|do|does|did|would|should) ([a-zA-Z\s]+)",
            "quantity": r"how (many|much|long|far|often) ([a-zA-Z\s]+)",
            "possibility": r"(can|could|would|will|should) ([a-zA-Z\s]+)",
            "opinion": r"(what do you think|your opinion|do you believe) ([a-zA-Z\s]+)"
        }

    def categorize_query(self, query):
        query = query.lower()
        best_match = (None, None, 0)  # (category, subcategory, match_count)
        
        for category, subcats in self.categories.items():
            for subcat, keywords in subcats.items():
                match_count = 0
                for keyword in keywords:
                    if keyword in query:
                        match_count += 1
                if match_count > best_match[2]:
                    best_match = (category, subcat, match_count)
        
        return best_match[0], best_match[1]

    def extract_question_type(self, query):
        query = query.lower()
        best_match = None
        longest_match = 0
        
        for q_type, pattern in self.knowledge_patterns.items():
            match = re.search(pattern, query)
            if match:
                match_length = len(match.group(0))
                if match_length > longest_match:
                    longest_match = match_length
                    best_match = (q_type, match.group(len(match.groups())))
        
        return best_match if best_match else (None, None)

    def get_all_keywords(self):
        keywords = set()
        for category in self.categories.values():
            for subcategory in category.values():
                keywords.update(subcategory)
        return keywords

class ChatBot:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='GeneralKnowledgeChatbot/1.0'
        )
        self.knowledge_base = KnowledgeBase()
        self.lemmatizer = WordNetLemmatizer()
        self.model = model
        
        # Enhanced greetings and responses
        self.greetings = {
            "hi": [
                "👋 Hi! I'm excited to help you learn! What's on your mind?",
                "✨ Hi there! Ready to explore some fascinating topics?",
                "🌟 Hi! I'm your friendly AI assistant. What would you like to know?",
                "💫 Hi! Let's discover something interesting together!",
                "🎯 Hi! Your curiosity is my inspiration. What shall we learn about?"
            ],
            "hello": [
                "👋 Hi there! I'm excited to chat with you! What would you like to know about?",
                "✨ Hello! Ready to explore the world of knowledge together?",
                "🌟 Hey! I'm your friendly knowledge companion. What interests you?",
                "💫 Greetings! Let's discover something fascinating today!",
                "🎯 Hi! I'm here to help you learn. What's on your mind?"
            ],
            "good morning": [
                "🌅 Good morning! Ready to start the day with some learning?",
                "🌄 Morning! Your curiosity brightens my day. What shall we explore?",
                "☀️ Good morning! Fresh day, fresh knowledge. What interests you?",
                "🌞 Morning! Let's make this a day of discovery!",
                "✨ Good morning! Your daily dose of knowledge awaits!"
            ],
            "good afternoon": [
                "🌞 Good afternoon! Perfect time for learning something new!",
                "⭐ Afternoon! What would you like to discover today?",
                "🌟 Good afternoon! Ready for some fascinating insights?",
                "💫 Afternoon! Let's explore something interesting!",
                "✨ Good afternoon! Your curiosity is welcome here!"
            ],
            "good evening": [
                "🌙 Good evening! The night is young for learning!",
                "🌠 Evening! Ready to explore some fascinating topics?",
                "✨ Good evening! What would you like to learn about?",
                "⭐ Evening greetings! Let's discover something interesting!",
                "💫 Good evening! Curiosity never sleeps!"
            ],
            "how are you": [
                "🌟 I'm fantastic and ready to share knowledge with you!",
                "✨ Doing great! Excited to help you learn something new!",
                "💫 I'm wonderful! Can't wait to explore topics with you!",
                "⭐ I'm excellent! Ready to answer your questions!",
                "🎯 I'm perfect! Looking forward to our learning journey!"
            ],
            "what is your name": [
                "🤖 I'm KnowBot, your personal knowledge companion!",
                "✨ KnowBot here! Ready to explore and learn together!",
                "🌟 I'm KnowBot, your friendly AI learning assistant!",
                "💫 Call me KnowBot! I'm here to help you discover!",
                "⭐ KnowBot at your service! Let's learn something new!"
            ],
            "who created you": [
                "🎯 I'm an AI created to help you explore our fascinating world!",
                "🤖 I'm your AI companion, designed to share knowledge!",
                "✨ I'm a friendly AI assistant, here to help you learn!",
                "🌟 I was created to make learning fun and interactive!",
                "💫 I'm an AI dedicated to sharing knowledge with you!"
            ],
            "bye": [
                "👋 Goodbye! Come back when curiosity strikes again!",
                "✨ See you later! Keep that learning spirit alive!",
                "🌟 Farewell! Remember, knowledge is an endless adventure!",
                "💫 Bye for now! Looking forward to our next chat!",
                "⭐ Take care! Come back soon for more discoveries!"
            ],
            "thank": [
                "✨ You're welcome! Your curiosity makes my day!",
                "🌟 Glad I could help! Keep those questions coming!",
                "💫 My pleasure! Learning together is what I'm here for!",
                "⭐ Anytime! Your enthusiasm for learning is wonderful!",
                "🎯 You're welcome! Keep exploring and learning!"
            ],
            "help": [
                "🌟 I can help you learn about any topic! Just ask away!",
                "✨ Ask me anything! I'm here to share knowledge!",
                "💫 Need information? I'm your go-to source!",
                "⭐ I can answer questions about science, history, arts, and more!",
                "🎯 How can I assist your learning journey today?"
            ],
            "default_positive": [
                "✨ That's interesting! Tell me more!",
                "🌟 I love your enthusiasm! What else interests you?",
                "💫 Great question! Let's explore that together!",
                "⭐ Fascinating topic! What would you like to know?",
                "🎯 I'm excited to help you learn about this!"
            ],
            "default_negative": [
                "💫 Don't worry, we'll figure this out together!",
                "✨ Let me try to help you in a different way!",
                "🌟 Perhaps we can approach this from another angle?",
                "⭐ Let's break this down and try again!",
                "🎯 I'm here to help! Let's try rephrasing that!"
            ],
            "default_neutral": [
                "🤔 Interesting! Could you tell me more about what you'd like to know?",
                "💭 I'm intrigued! What specific aspect interests you?",
                "✨ Let's explore this together! What would you like to focus on?",
                "🌟 I'm here to help! Could you elaborate a bit more?",
                "💫 That's a great topic! What would you like to learn about it?"
            ],
            "error": [
                "💫 I'm not quite sure about that. Could you rephrase your question?",
                "✨ I didn't find specific information about that. Could you try asking differently?",
                "🌟 I'd love to help! Could you rephrase that for me?",
                "⭐ Hmm, I'm not finding what you're looking for. Could you try another way?",
                "🎯 Let's try that again with different wording!"
            ]
        }

        # Response variations for knowledge answers
        self.response_prefixes = {
            "knowledge": [
                "📚 Here's what I found: ",
                "💡 Let me share this with you: ",
                "🔍 According to my knowledge: ",
                "📖 Here's some information: ",
                "🎓 Here's what I know: "
            ],
            "explanation": [
                "🌟 Let me explain: ",
                "✨ Here's how it works: ",
                "💫 To understand this better: ",
                "⭐ Here's the explanation: ",
                "🎯 Let me break this down: "
            ],
            "definition": [
                "📝 By definition: ",
                "📚 This refers to: ",
                "💡 To put it simply: ",
                "🔍 This means: ",
                "📖 In essence: "
            ],
            "fact": [
                "⚡ Fun fact: ",
                "🌟 Interesting fact: ",
                "✨ Did you know: ",
                "💫 Here's a fascinating detail: ",
                "⭐ Notable fact: "
            ],
            "error": [
                "💫 I'm not quite sure about that. Could you rephrase your question?",
                "✨ I didn't find specific information about that. Could you try asking differently?",
                "🌟 I'd love to help! Could you rephrase that for me?",
                "⭐ Hmm, I'm not finding what you're looking for. Could you try another way?",
                "🎯 Let's try that again with different wording!"
            ]
        }

    def correct_spelling(self, text):
        try:
            # Split text into words
            words = text.split()
            corrected_words = []
            
            for word in words:
                # Skip short words, numbers, and special characters
                if len(word) <= 2 or word.isnumeric() or not word.isalnum():
                    corrected_words.append(word)
                    continue
                
                # Use TextBlob for spell checking
                w = Word(word)
                correction = w.correct()
                
                # If correction confidence is low, use difflib
                if correction == word:
                    possible_words = difflib.get_close_matches(word, 
                                                             self.knowledge_base.get_all_keywords(), 
                                                             n=1, 
                                                             cutoff=0.7)
                    if possible_words:
                        correction = possible_words[0]
                
                corrected_words.append(correction)
            
            return ' '.join(corrected_words)
        except Exception as e:
            print(f"Spell correction error: {e}")
            return text

    def get_wiki_summary(self, topic, sentences=3):
        try:
            # Lemmatize and clean the topic for better search results
            words = word_tokenize(topic)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            search_topic = ' '.join(lemmatized_words)
            
            # Try exact match first
            page = self.wiki.page(search_topic)
            if not page.exists():
                # If no exact match, try capitalizing first letters
                search_topic = ' '.join(word.capitalize() for word in search_topic.split())
                page = self.wiki.page(search_topic)
            
            if page.exists():
                # Get full summary and split into sentences
                all_sentences = sent_tokenize(page.summary)
                
                # Select most relevant sentences
                if len(all_sentences) > sentences:
                    selected_sentences = all_sentences[:sentences]
                else:
                    selected_sentences = all_sentences
                
                summary = ' '.join(selected_sentences)
                return summary + " 📚"
            return None
        except Exception as e:
            print(f"Error in wiki search: {e}")
            return None

    def extract_topic(self, text):
        # Remove stopwords and get important words
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        important_words = [self.lemmatizer.lemmatize(word) 
                         for word in words 
                         if word not in stop_words and word.isalnum()]
        return ' '.join(important_words)

    def get_gemini_response(self, query):
        try:
            if self.model:
                response = self.model.generate_content(query)
                return response.text + " 🤖"
            return None
        except Exception as e:
            print(f"Error in Gemini response: {e}")
            return None

    def get_response(self, user_input):
        # Check for greetings first
        user_input_lower = user_input.lower().strip()
        
        # Enhanced greeting detection
        greeting_words = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        
        # Check for exact matches first
        for key, responses in self.greetings.items():
            if user_input_lower == key:
                return random.choice(responses)
        
        # Then check for contained greetings
        for word in greeting_words:
            if word in user_input_lower:
                key = word if word in self.greetings else "hello"
                return random.choice(self.greetings[key])
        
        # Correct spelling mistakes
        corrected_input = self.correct_spelling(user_input)
        if corrected_input != user_input:
            print(f"Corrected: {corrected_input}")
        
        # Identify the type of question and category
        question_type, topic = self.knowledge_base.extract_question_type(corrected_input)
        category, subcategory = self.knowledge_base.categorize_query(corrected_input)
        
        # If it's a knowledge question
        if any(phrase in corrected_input.lower() for phrase in ["what", "who", "tell", "explain", "how", "where", "when", 
                                                               "why", "which", "define", "describe", "can you", "could you",
                                                               "would you", "should", "does", "do", "did", "is", "are"]):
            # Try Gemini first
            gemini_response = self.get_gemini_response(corrected_input)
            if gemini_response:
                # Add some variety to the response
                prefix = random.choice(self.response_prefixes["knowledge"])
                
                context_parts = []
                if category and subcategory:
                    context_parts.append(f"[{category.title()} - {subcategory.title()}]")
                if question_type:
                    context_parts.append(f"[{question_type.title()}]")
                
                context = " ".join(context_parts) + " " if context_parts else ""
                return context + prefix + gemini_response

            # Extract the topic
            if not topic:
                topic = self.extract_topic(corrected_input)
            
            # Fallback to Wikipedia
            wiki_response = self.get_wiki_summary(topic)
            
            if wiki_response:
                # Add some variety to the response
                prefix = random.choice(self.response_prefixes["knowledge"])
                
                context_parts = []
                if category and subcategory:
                    context_parts.append(f"[{category.title()} - {subcategory.title()}]")
                if question_type:
                    context_parts.append(f"[{question_type.title()}]")
                
                context = " ".join(context_parts) + " " if context_parts else ""
                return context + prefix + wiki_response + " ✨"
            
            return random.choice(self.response_prefixes["error"])
        
        # Analyze sentiment for non-knowledge questions
        sentiment = TextBlob(corrected_input).sentiment.polarity
        if sentiment > 0.5:
            return random.choice(self.response_prefixes["default_positive"])
        elif sentiment < -0.5:
            return random.choice(self.response_prefixes["default_negative"])
        
        return random.choice(self.response_prefixes["default_neutral"])

class ChatBubble(BoxLayout):
    message = StringProperty('')
    is_user = ObjectProperty(False)
    opacity = NumericProperty(0)

    def on_size(self, *args):
        self.opacity = 0
        Animation(opacity=1, duration=0.3).start(self)

class ChatInterface(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chatbot = ChatBot()
        Window.bind(on_resize=self.on_window_resize)
        Clock.schedule_once(lambda dt: setattr(self.ids.message_input, 'focus', True), 0.1)
        Window.clearcolor = (0.05, 0.05, 0.07, 1)
        
        # Add initial greeting
        Clock.schedule_once(lambda dt: self.add_message(
            random.choice(self.chatbot.greetings["hello"])
        ), 1)
        
    def on_window_resize(self, instance, width, height):
        self.ids.chat_scroll.height = height - dp(150)  # Adjust scroll view height

    def add_message(self, message, is_user=False):
        chat_bubble = ChatBubble(message=message, is_user=is_user)
        self.ids.chat_layout.add_widget(chat_bubble)
        
        # Animate the bubble
        anim = Animation(opacity=1, duration=0.3)
        anim.start(chat_bubble)
        
        # Schedule scrolling after the bubble is added
        Clock.schedule_once(self.scroll_to_bottom, 0.1)

    def scroll_to_bottom(self, dt):
        self.ids.chat_scroll.scroll_y = 0

    def send_message(self, *args):
        message = self.ids.message_input.text.strip()
        if message:
            # Add user message
            self.add_message(message, is_user=True)
            
            # Clear input
            self.ids.message_input.text = ''
            
            # Get and display bot response
            def get_response():
                response = self.chatbot.get_response(message)
                Clock.schedule_once(lambda dt: self.add_message(response), 1)
            
            # Start response in a thread to prevent UI blocking
            threading.Thread(target=get_response).start()

class ChatBotApp(App):
    def build(self):
        # Set window size and minimum size
        Window.size = (400, 700)
        Window.minimum_width = 300
        Window.minimum_height = 500
        
        # Create interface
        interface = ChatInterface()
        
        # Add window animations
        def animate_window(dt):
            interface.opacity = 0
            anim = Animation(opacity=1, duration=0.5)
            anim.start(interface)
        Clock.schedule_once(animate_window)
        
        return interface

if __name__ == '__main__':
    ChatBotApp().run() 