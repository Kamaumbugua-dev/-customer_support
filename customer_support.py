import streamlit as st
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="AI Customer Support Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Training data - Questions and Answers
TRAINING_DATA = [
    # Greetings
    ("Hello", "Hi there!  Welcome to our Customer Support. I'm your AI assistant, ready to help you 24/7. What can I do for you today?"),
    ("Hi", "Hello! Thanks for reaching out. How can I assist you today?"),
    ("Hey there", "Hey!  Great to see you. What would you like help with?"),
    ("Good morning", "Good morning!  How can I make your day better?"),
    
    # Order Status
    ("Where is my order", " I can help you track your order! To check your order status:\n\n1. Log into your account\n2. Go to 'My Orders'\n3. Click on the specific order\n\nYou'll see real-time tracking. Orders typically arrive in 3-5 business days. Do you have your order number handy?"),
    ("Track my order", " To track your order:\n• Visit 'My Orders' in your account\n• Select the order you want to track\n• View real-time delivery updates\n\nNeed help finding a specific order? Share your order number!"),
    ("When will my order arrive", " Delivery times depend on your shipping method:\n• Standard: 3-5 business days\n• Express: 1-2 business days\n\nYou can track your order in real-time through 'My Orders'. What's your order number?"),
    ("Order status check", " Let me help you check your order status! Please log into your account and visit 'My Orders' section. You'll find detailed tracking information there. Need specific help with an order?"),
    
    # Returns & Refunds
    ("How do I return an item", "↩ Our return process is simple:\n\n1. Go to 'My Orders'\n2. Select the item to return\n3. Click 'Return Item'\n4. Choose reason and submit\n\nReturns accepted within 30 days. Items must be unused with original packaging. Refunds process in 7-10 days."),
    ("What is your return policy", "↩ Our Return Policy:\n• 30-day return window from delivery\n• Items must be unused and in original packaging\n• Free return shipping on defective items\n• Refunds processed within 7-10 business days\n\nWant to start a return?"),
    ("Refund policy", " Refund Policy:\n• Refunds issued to original payment method\n• Processing time: 7-10 business days\n• You'll receive email confirmation\n• Store credit option available for faster processing\n\nNeed to request a refund?"),
    ("Can I exchange a product", " Yes! Exchanges are easy:\n1. Initiate return for the item\n2. Select 'Exchange' option\n3. Choose replacement product\n4. We'll ship the new item once we receive the return\n\nWant to start an exchange?"),
    
    # Payment Methods
    ("What payment methods do you accept", " We accept multiple payment options:\n\n• Credit/Debit Cards (Visa, Mastercard, Amex, Discover)\n• UPI (Google Pay, PhonePe, Paytm)\n• Net Banking\n• Digital Wallets (PayPal, Apple Pay)\n• Cash on Delivery (COD)\n\nAll transactions are secured with 256-bit encryption. "),
    ("Do you accept credit cards", " Absolutely! We accept all major credit cards:\n• Visa\n• Mastercard\n• American Express\n• Discover\n\nYour payment information is always secure with us. "),
    ("Payment options available", " Multiple payment options for your convenience:\n• Cards (Credit/Debit)\n• UPI payments\n• Net Banking\n• Digital Wallets\n• Cash on Delivery\n\nAll payments are 100% secure!"),
    ("Is payment secure", " Yes, absolutely! Your security is our priority:\n• 256-bit SSL encryption\n• PCI DSS compliant\n• No card details stored on our servers\n• Secure payment gateway partners\n\nYou can shop with confidence!"),
    
    # Shipping
    ("What are the shipping charges", " Shipping Charges:\n• FREE shipping on orders above $50\n• Standard shipping: $5 (3-5 business days)\n• Express shipping: $15 (1-2 business days)\n• International: Calculated at checkout\n\nCharges vary by location and weight."),
    ("Free shipping", " Yes! We offer FREE shipping on all orders above $50. For orders below $50:\n• Standard: $5\n• Express: $15\n\nAdd items to reach $50 and enjoy free delivery!"),
    ("How long does shipping take", " Shipping Timeframes:\n• Standard: 3-5 business days\n• Express: 1-2 business days\n• International: 7-14 business days\n\nYou'll receive tracking information via email once your order ships!"),
    
    # Cancel Order
    ("How to cancel my order", " To cancel your order:\n\n1. Log into your account\n2. Go to 'My Orders'\n3. Find the order\n4. Click 'Cancel Order'\n\n Note: Orders can only be cancelled before shipping. Once shipped, you'll need to use our return process. What's your order number?"),
    ("Cancel order", " I can help you cancel your order! Here's how:\n• Visit 'My Orders' section\n• Select the order to cancel\n• Click 'Cancel Order' button\n\nRemember: Cancellation only works before the item ships. Need immediate help? Share your order number."),
    ("Stop my order", " To stop your order, you need to cancel it quickly before it ships:\n1. Go to 'My Orders'\n2. Click 'Cancel Order'\n\nIf already shipped, you can refuse delivery or initiate a return. What's your order status?"),
    
    # Product Information
    ("Tell me about your products", " We offer a wide range of products!\n\nFor detailed information:\n• Visit product pages for specifications\n• Check customer reviews and ratings\n• View images and videos\n• Compare similar products\n\nWhat category interests you?"),
    ("Product details", " Looking for product details? I can help!\n\nYou can find:\n• Full specifications\n• Customer reviews\n• High-quality images\n• Video demonstrations\n• Size guides and comparisons\n\nWhich product would you like to know about?"),
    ("Do you have product reviews", " Yes! Every product has customer reviews:\n• Verified purchase reviews\n• Star ratings\n• Photos from customers\n• Helpful vote system\n\nReviews help you make informed decisions. What product are you interested in?"),
    
    # Contact Support
    ("How can I contact support", " Multiple ways to reach us:\n\n• Phone: +254702569778 (24/7)\n• Email: stevenk710@gmail.com\n• Live Chat: Available on website\n• Social Media: @customersupport\n\nResponse time: Within 24 hours\n\nWould you like me to connect you with a human agent?"),
    ("Talk to a human", " I'll connect you with our human support team!\n\n• Phone: +254702569778 (immediate)\n• Live Chat: Click chat icon (2-3 min wait)\n• Email: stevenk710@gmail.com (24h response)\n\nWhat's the best way to reach you?"),
    ("Customer service number", " Our customer service:\n• Phone: 254702569778\n• Available 24/7\n• Average wait time: 3-5 minutes\n\nYou can also use live chat on our website for instant help!"),
    
    # Account Issues
    ("I forgot my password", " No worries! Reset your password:\n\n1. Click 'Forgot Password' on login page\n2. Enter your email\n3. Check email for reset link\n4. Create new password\n\nNot receiving the email? Check spam folder or contact support."),
    ("How do I create an account", " Creating an account is easy:\n\n1. Click 'Sign Up'\n2. Enter email and password\n3. Verify email\n4. Complete profile\n\nBenefits: Order tracking, faster checkout, exclusive deals!"),
    ("Update my account information", " To update your account:\n\n1. Log in to your account\n2. Go to 'Account Settings'\n3. Edit information (name, email, address)\n4. Click 'Save Changes'\n\nNeed help with specific details?"),
    
    # Promotions & Deals
    ("Do you have any discounts", " Yes! Current offers:\n• Sign up: 10% off first order\n• Newsletter: Exclusive deals\n• Seasonal sales: Up to 50% off\n\nCheck our 'Deals' section for latest promotions. Want to subscribe for updates?"),
    ("Current sales", " Ongoing Sales:\n• Weekly deals section\n• Flash sales (limited time)\n• Clearance items up to 70% off\n\nVisit our homepage for today's featured deals!"),
    ("Coupon codes", " Coupon codes:\n• First-time users: WELCOME10 (10% off)\n• Newsletter subscribers get exclusive codes\n• Check email for personalized offers\n\nApply codes at checkout. Happy shopping!"),
    
    # Delivery Issues
    ("My order is late", " Sorry your order is delayed! Let me help:\n\n1. Check tracking for latest update\n2. Delays can occur due to weather/holidays\n3. If beyond estimated date, contact support\n\nWhat's your order number? I'll look into it."),
    ("Package damaged", " Sorry about the damaged package!\n\n1. Take photos of damage\n2. Go to 'My Orders'\n3. Report issue with photos\n4. We'll arrange replacement/refund\n\nWe'll resolve this quickly for you!"),
    ("Wrong item received", " Apologies for the mix-up!\n\n1. Don't open/use the item\n2. Go to 'My Orders'\n3. Select 'Wrong Item Received'\n4. We'll send correct item + return label\n\nNo charge for our mistake!"),
]

class SmartChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
        self.questions = [q for q, a in TRAINING_DATA]
        self.answers = [a for q, a in TRAINING_DATA]
        self.question_vectors = None
        self.train()
    
    def train(self):
        """Train the chatbot using TF-IDF vectorization"""
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
    
    def get_response(self, user_query, threshold=0.3):
        """Get response using semantic similarity"""
        # Vectorize user query
        query_vector = self.vectorizer.transform([user_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        # Get best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Return answer if similarity is above threshold
        if best_similarity > threshold:
            return self.answers[best_match_idx], best_similarity
        else:
            return self.get_fallback_response(), 0.0
    
    def get_fallback_response(self):
        """Fallback response when no good match is found"""
        return """I'm not entirely sure about that.  Let me help you with common topics:

 **Orders**: "Where is my order?" | "Cancel my order"
↩ **Returns**: "How to return?" | "Refund policy"
 **Payments**: "Payment methods" | "Is payment secure?"
 **Shipping**: "Shipping charges" | "Free shipping"
 **Products**: "Product details" | "Reviews"
 **Contact**: "Talk to human" | "Customer service"

Please rephrase your question or choose a topic above!"""

# Initialize chatbot
@st.cache_resource
def load_chatbot():
    return SmartChatbot()

chatbot = load_chatbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello!  I'm your AI-powered customer support assistant. I use machine learning to understand your questions better. How can I help you today?",
        "timestamp": datetime.now().strftime("%H:%M"),
        "confidence": 1.0
    })

# App header
st.title(" AI Customer Support Chatbot")
st.markdown("*Powered by Machine Learning & Semantic Search*")
st.divider()

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(message["timestamp"])
            with col2:
                if message["role"] == "assistant" and "confidence" in message:
                    confidence = message["confidence"]
                    if confidence > 0:
                        st.caption(f" {confidence:.0%}")

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Get bot response using ML
    bot_response, confidence = chatbot.get_response(user_input)
    
    # Add bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response,
        "timestamp": datetime.now().strftime("%H:%M"),
        "confidence": confidence
    })
    
    # Rerun to update chat
    st.rerun()

# Sidebar with information
with st.sidebar:
    st.header(" About This AI Chatbot")
    st.markdown("""
    This chatbot uses **Machine Learning** with:
    
    -  **TF-IDF Vectorization**: Converts text to numerical vectors
    -  **Cosine Similarity**: Measures semantic similarity
    -  **Confidence Scores**: Shows how confident the AI is
    
    **Advantages over rule-based:**
    - Understands variations in questions
    - No need for exact keyword matches
    - Handles typos and different phrasings
    - Learns from training data
    """)
    
    st.divider()
    
    st.header(" Try These Questions:")
    example_questions = [
        "Where's my package?",
        "I want to return something",
        "What payment do you take?",
        "How much is shipping?",
        "Need to cancel my order",
        "Is my payment safe?",
        "Talk to a person"
    ]
    
    for question in example_questions:
        if st.button(question, key=question, use_container_width=True):
            # Simulate user asking this question
            st.session_state.messages.append({
                "role": "user",
                "content": question,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            bot_response, confidence = chatbot.get_response(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "timestamp": datetime.now().strftime("%H:%M"),
                "confidence": confidence
            })
            st.rerun()
    
    st.divider()
    
    if st.button(" Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello!  I'm your AI-powered customer support assistant. I use machine learning to understand your questions better. How can I help you today?",
            "timestamp": datetime.now().strftime("%H:%M"),
            "confidence": 1.0
        })
        st.rerun()
    
    st.divider()
    st.caption(" ML-Powered Chatbot")
    st.caption(" Semantic Search Enabled")
    st.caption(" Available 24/7")