# -customer_support

#  AI-Powered Customer Support Chatbot

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A machine learning-powered chatbot that understands customer queries using semantic search and provides intelligent responses 24/7.



##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Training Data](#training-data)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

##  Overview

This project demonstrates an AI-powered customer support chatbot that uses **Natural Language Processing (NLP)** and **Machine Learning** to understand and respond to customer queries. Unlike traditional rule-based chatbots, this system uses semantic search to match user intent, making it more flexible and intelligent.

### Why This Project?

- **Real-world Application**: Used by companies like Amazon, Flipkart, and Zomato
- **Scalable Solution**: Handles multiple queries simultaneously
- **Cost-Effective**: Reduces customer support workload by 60-80%
- **Always Available**: 24/7 automated customer service

##  Features

### Core Capabilities
-  **Semantic Understanding** - Understands variations in user questions
-  **Confidence Scoring** - Displays AI confidence for each response
-  **Multi-Topic Support** - Handles 10+ customer service categories
-  **Real-time Responses** - Instant answers without delays
-  **Fallback Handling** - Smart responses when uncertain

### Technical Features
- **TF-IDF Vectorization** for text-to-numerical conversion
- **Cosine Similarity** for semantic matching
- **Threshold-based responses** (30% minimum confidence)
- **Interactive web interface** built with Streamlit
- **Session management** for conversation history

##  Demo

### Live Demo
 **[Try the live chatbot here](your-streamlit-app-url)** *(Deploy and add link)*

### Example Interactions

```
User: "Where's my package?"
Bot:  I can help you track your order!
     Confidence: 87%

User: "I want my money back"
Bot:  Our Return Policy: You can return items within 30 days...
     Confidence: 82%

User: "Is payment safe?"
Bot:  Yes, absolutely! Your security is our priority...
     Confidence: 91%
```

##  How It Works

### Architecture Flow

```
User Input → Preprocessing → TF-IDF Vectorization → Cosine Similarity → Best Match → Response
```

### Technical Process

1. **Training Phase**
   - Load 50+ question-answer pairs
   - Apply TF-IDF vectorization to questions
   - Create vector representations of all training questions

2. **Query Phase**
   - User inputs a question
   - Question is vectorized using the same TF-IDF model
   - Cosine similarity is calculated against all training vectors
   - Best match is selected if confidence > 30%
   - Appropriate response is returned

3. **Fallback Mechanism**
   - If confidence < 30%, fallback response is triggered
   - Suggests common topics and rephrasing

### Mathematical Foundation

**TF-IDF (Term Frequency-Inverse Document Frequency)**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
where:
- TF(t,d) = frequency of term t in document d
- IDF(t) = log(N / df(t))
- N = total number of documents
- df(t) = number of documents containing term t
```

**Cosine Similarity**
```
similarity(A,B) = cos(θ) = (A·B) / (||A|| ||B||)
```

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/kamaumbugua-dev/ai-customer_support.git
cd ai-customer_support
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run chatbot_app.py
```

5. **Open in browser**
- The app will automatically open at `http://localhost:8501`
- If not, manually navigate to the URL shown in terminal

##  Usage

### Basic Usage

```python
# Import the chatbot class
from chatbot_app import SmartChatbot

# Initialize chatbot
bot = SmartChatbot()

# Get response
response, confidence = bot.get_response("Where is my order?")
print(f"Response: {response}")
print(f"Confidence: {confidence:.2%}")
```

### Adding New Training Data

```python
# Add to TRAINING_DATA list in chatbot_app.py
TRAINING_DATA.append(
    ("Your new question", "Your answer to this question")
)
```

### Adjusting Confidence Threshold

```python
# In get_response method, modify threshold parameter
response, confidence = bot.get_response(user_query, threshold=0.4)
```

## Project Structure

```
ai-customer-chatbot/
│
├── chatbot_app.py          # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT License
│
├── data/                  # (Optional) External training data
│   └── faq_data.csv
│
├── models/                # (Optional) Saved models
│   └── chatbot_model.pkl
│
├── tests/                 # Unit tests
│   └── test_chatbot.py
│
└── screenshots/           # Demo images
    ├── demo.png
    └── architecture.png
```

##  Technologies Used

### Core Technologies
- **Python 3.8+** - Programming language
- **Streamlit** - Web framework for ML applications
- **scikit-learn** - Machine learning library
- **NumPy** - Numerical computations

### ML/NLP Components
- **TfidfVectorizer** - Text vectorization
- **cosine_similarity** - Similarity measurement

### Additional Libraries
```python
streamlit==1.28.0
scikit-learn==1.3.0
numpy==1.24.3
```

##  Training Data

The chatbot is trained on 50+ carefully curated question-answer pairs covering:

| Category | Examples | Count |
|----------|----------|-------|
|  Greetings | "Hello", "Hi there" | 4 |
|  Order Status | "Where is my order?", "Track package" | 6 |
|  Returns | "How to return?", "Refund policy" | 5 |
|  Payments | "Payment methods", "Is payment secure?" | 4 |
|  Shipping | "Shipping charges", "Free shipping" | 4 |
|  Cancellations | "Cancel order", "Stop my order" | 3 |
|  Products | "Product details", "Reviews" | 3 |
|  Contact | "Talk to human", "Customer service" | 3 |
|  Account | "Forgot password", "Create account" | 3 |
|  Promotions | "Discounts", "Coupon codes" | 3 |
|  Delivery Issues | "Late order", "Damaged package" | 3 |

### Data Format
```python
("User question variant", "Detailed response with helpful information")
```

##  Future Enhancements

### Planned Features
- [ ] **Deep Learning Integration** - BERT/GPT models for better understanding
- [ ] **Multi-language Support** - Support for Spanish, French, Hindi, etc.
- [ ] **Voice Input/Output** - Speech-to-text and text-to-speech
- [ ] **Sentiment Analysis** - Detect customer emotions
- [ ] **Analytics Dashboard** - Track common queries and satisfaction
- [ ] **Database Integration** - Real-time order tracking
- [ ] **User Authentication** - Personalized responses
- [ ] **Live Chat Handoff** - Seamless transfer to human agents
- [ ] **A/B Testing** - Compare different response strategies
- [ ] **Feedback Loop** - Learn from user corrections

### Advanced ML Features
- Intent classification with neural networks
- Named Entity Recognition (NER) for extracting order numbers, dates
- Context-aware conversations with memory
- Reinforcement learning from user feedback

##  Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation as needed
- Keep commits atomic and well-described

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Contact

**Your Name**
- GitHub: [@kamaumbugua-dev](https://github.com/kamaumbugua-dev)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/steven-kamau-mbugua)
- Email: stevokama45@gmail.com
- Portfolio: [yourportfolio.com](https://yourportfolio.com)

##  Acknowledgments

- Inspired by customer support systems at Amazon, Flipkart, and Zomato
- Customer Support Chat Dataset from [Kaggle] 
- Streamlit community for excellent documentation
- scikit-learn team for powerful ML tools

---

##  Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

[![Star History Chart](https://api.star-history.com/svg?repos=kamaumbugua-dev/customer_support&type=Date)](https://star-history.com/#kamaumbugua_dev/customer_support&Date)

---

**Made with  and  by Steven Kamau Mbugua**

*Last updated: January 2026*
