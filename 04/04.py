"""
=============================================================================
  AI FOR BUSINESS — Week 04
  Topic : AI & Marketing — Personalized Content Recommendation System
  File  : 04.py

  Approach : Hybrid Recommendation Engine
    1. Content-Based Filtering  (TF-IDF cosine similarity)
    2. Collaborative Filtering  (user-item matrix cosine similarity)
    3. Hybrid Scoring            final = α·content + (1-α)·collab

  No external API keys required.
  Dependencies : numpy, scikit-learn  (pip install numpy scikit-learn)
=============================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import sys
import math
import copy
import textwrap

# Ensure Unicode output works on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("\n[ERROR] Missing dependencies.  Please run:\n"
          "        pip install numpy scikit-learn\n")
    raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  CONTENT CATALOG
#      30 website articles across 6 categories
# ═══════════════════════════════════════════════════════════════════════════════

ARTICLES = [
    # ── Technology ─────────────────────────────────────────────────────────────
    {
        "id": "A001", "category": "Technology",
        "title": "The Rise of Quantum Computing",
        "tags": ["quantum", "computing", "hardware", "future"],
        "text": (
            "Quantum computing leverages quantum bits or qubits to process information "
            "in ways classical computers cannot. Companies like IBM, Google, and startups "
            "are racing to achieve quantum advantage. Applications span drug discovery, "
            "cryptography, optimization, and machine learning."
        ),
    },
    {
        "id": "A002", "category": "Technology",
        "title": "5G Networks and the Internet of Things",
        "tags": ["5G", "IoT", "connectivity", "networks", "mobile"],
        "text": (
            "Fifth-generation wireless technology promises ultra-low latency and massive "
            "device connectivity. IoT sensors, autonomous vehicles, smart factories, and "
            "telemedicine are among the key beneficiaries. Telecom giants are investing "
            "billions in 5G infrastructure worldwide."
        ),
    },
    {
        "id": "A003", "category": "Technology",
        "title": "Cloud Computing: Trends for 2025",
        "tags": ["cloud", "AWS", "Azure", "SaaS", "serverless"],
        "text": (
            "Cloud adoption continues to accelerate. Multi-cloud strategies, serverless "
            "architectures, and edge computing are dominant trends. Enterprises are "
            "migrating legacy workloads while adopting DevSecOps practices. Cost "
            "optimisation and sustainability are emerging priorities."
        ),
    },
    {
        "id": "A004", "category": "Technology",
        "title": "Cybersecurity Threats in the Age of AI",
        "tags": ["cybersecurity", "AI", "hacking", "defence", "data breach"],
        "text": (
            "AI-powered cyberattacks, deepfake phishing, and ransomware-as-a-service are "
            "reshaping the threat landscape. Defenders use AI for anomaly detection and "
            "automated incident response. Zero-trust architecture and supply-chain "
            "security are becoming boardroom conversations."
        ),
    },
    {
        "id": "A005", "category": "Technology",
        "title": "Open Source Software: Power of the Community",
        "tags": ["open source", "Linux", "GitHub", "community", "development"],
        "text": (
            "Open-source projects power the majority of today's digital infrastructure. "
            "Linux, Kubernetes, Python, and React are community-driven successes. "
            "Enterprises contribute to open source for talent attraction and influence. "
            "Licensing, governance, and sustainability remain ongoing challenges."
        ),
    },
    # ── Artificial Intelligence ────────────────────────────────────────────────
    {
        "id": "A006", "category": "AI",
        "title": "Generative AI: Beyond ChatGPT",
        "tags": ["generative AI", "LLM", "GPT", "creativity", "content"],
        "text": (
            "Large language models and image generators are transforming creative "
            "industries. Businesses use generative AI for copywriting, code generation, "
            "customer service, and product design. Ethical concerns around bias, "
            "intellectual property, and misinformation are intensifying."
        ),
    },
    {
        "id": "A007", "category": "AI",
        "title": "Machine Learning in Predictive Analytics",
        "tags": ["machine learning", "predictive analytics", "data science", "business"],
        "text": (
            "Predictive models analyse historical data to forecast demand, churn, fraud, "
            "and maintenance failures. Decision trees, random forests, and gradient "
            "boosting are workhorses of enterprise analytics. Explainability and data "
            "quality remain key implementation hurdles."
        ),
    },
    {
        "id": "A008", "category": "AI",
        "title": "Computer Vision Applications in Retail",
        "tags": ["computer vision", "retail", "object detection", "automation"],
        "text": (
            "Retailers deploy computer vision for shelf monitoring, checkout-free stores, "
            "customer behaviour analytics, and loss prevention. Deep learning models "
            "like YOLO and Faster R-CNN power real-time object detection. Privacy "
            "regulations shape deployment strategies."
        ),
    },
    {
        "id": "A009", "category": "AI",
        "title": "Natural Language Processing for Customer Support",
        "tags": ["NLP", "chatbot", "customer support", "sentiment analysis"],
        "text": (
            "NLP enables chatbots to handle queries, analyse sentiment, route tickets, "
            "and generate automatic replies. Intent recognition and entity extraction "
            "models improve first-contact resolution. Multilingual support and context "
            "retention are active research areas."
        ),
    },
    {
        "id": "A010", "category": "AI",
        "title": "Responsible AI: Ethics and Governance",
        "tags": ["responsible AI", "ethics", "bias", "fairness", "regulation"],
        "text": (
            "Governments and corporations are developing AI governance frameworks. "
            "Principles of fairness, accountability, transparency, and data privacy "
            "guide responsible deployment. The EU AI Act and similar legislation are "
            "setting global benchmarks for high-risk AI systems."
        ),
    },
    # ── Finance ────────────────────────────────────────────────────────────────
    {
        "id": "A011", "category": "Finance",
        "title": "Cryptocurrency Market: What's Next?",
        "tags": ["cryptocurrency", "Bitcoin", "blockchain", "DeFi", "investment"],
        "text": (
            "Bitcoin, Ethereum, and altcoins continue to attract retail and institutional "
            "investors. DeFi platforms offer lending, borrowing, and yield farming "
            "without intermediaries. Regulatory clarity, macroeconomic conditions, and "
            "adoption by payment networks drive sentiment."
        ),
    },
    {
        "id": "A012", "category": "Finance",
        "title": "AI-Driven Algorithmic Trading",
        "tags": ["algorithmic trading", "AI", "finance", "quantitative", "stock market"],
        "text": (
            "Quant funds apply deep learning and reinforcement learning to exploit market "
            "micro-structure inefficiencies. High-frequency trading, sentiment analysis "
            "from news feeds, and portfolio optimisation are mainstream. Regulators "
            "scrutinise flash crashes and systemic risk."
        ),
    },
    {
        "id": "A013", "category": "Finance",
        "title": "Personal Finance: Budgeting in a High-Inflation World",
        "tags": ["personal finance", "budgeting", "inflation", "savings", "investment"],
        "text": (
            "Rising prices erode purchasing power and demand smarter budgeting. Tools "
            "like zero-based budgeting, expense tracking apps, and automated savings help "
            "consumers adapt. Diversifying into equities, real assets, and I-bonds are "
            "popular inflation hedges."
        ),
    },
    {
        "id": "A014", "category": "Finance",
        "title": "ESG Investing: Profit Meets Purpose",
        "tags": ["ESG", "investing", "sustainability", "green finance", "impact"],
        "text": (
            "Environmental, Social, and Governance criteria guide trillions in assets. "
            "Investors demand transparent reporting, carbon reduction targets, and "
            "ethical supply chains. Greenwashing scandals and standardisation of ESG "
            "metrics are ongoing challenges for fund managers."
        ),
    },
    {
        "id": "A015", "category": "Finance",
        "title": "FinTech Innovation: Banking the Unbanked",
        "tags": ["fintech", "financial inclusion", "mobile banking", "micro-finance"],
        "text": (
            "Mobile money platforms and neo-banks serve populations excluded from "
            "traditional banking. AI-driven credit scoring from alternative data expands "
            "access. CBDC pilots by central banks signal deeper digitisation of the "
            "monetary system."
        ),
    },
    # ── Marketing ─────────────────────────────────────────────────────────────
    {
        "id": "A016", "category": "Marketing",
        "title": "Personalisation at Scale with AI",
        "tags": ["personalisation", "AI", "marketing", "e-commerce", "recommendation"],
        "text": (
            "AI enables dynamic personalisation of emails, landing pages, and product "
            "recommendations for millions of customers simultaneously. Machine learning "
            "models predict next-best-action, lifetime value, and churn risk. Privacy "
            "regulations and cookie deprecation demand first-party data strategies."
        ),
    },
    {
        "id": "A017", "category": "Marketing",
        "title": "Content Marketing Strategy for 2025",
        "tags": ["content marketing", "SEO", "blog", "video", "social media"],
        "text": (
            "High-quality, search-optimised content drives organic traffic and brand "
            "authority. Short-form video, interactive infographics, and podcasts dominate "
            "engagement. AI writing assistants accelerate production while human editors "
            "maintain quality and brand voice."
        ),
    },
    {
        "id": "A018", "category": "Marketing",
        "title": "Social Media Advertising: ROI Measurement",
        "tags": ["social media", "advertising", "ROI", "Facebook", "Instagram", "analytics"],
        "text": (
            "Attribution modelling across multiple social platforms challenges marketers. "
            "Meta, TikTok, LinkedIn, and Pinterest offer diverse ad formats and targeting. "
            "Brand safety, ad fraud detection, and cross-channel measurement require "
            "sophisticated analytics stacks."
        ),
    },
    {
        "id": "A019", "category": "Marketing",
        "title": "Email Marketing Automation Best Practices",
        "tags": ["email marketing", "automation", "CRM", "segmentation", "open rate"],
        "text": (
            "Behavioural triggers, A/B testing, and dynamic content personalisation lift "
            "open and click-through rates. Marketing automation platforms integrate with "
            "CRM to track lifecycle stages. AI-powered send-time optimisation and subject "
            "line generation improve campaign performance."
        ),
    },
    {
        "id": "A020", "category": "Marketing",
        "title": "Influencer Marketing: Authenticity vs. Scale",
        "tags": ["influencer", "marketing", "brand", "micro-influencer", "social proof"],
        "text": (
            "Micro-influencers with engaged niche audiences often outperform mega "
            "celebrities on ROI. Brands diversify across YouTube, Instagram, and TikTok. "
            "Performance tracking via affiliate links, promo codes, and UTM parameters "
            "is increasingly standard."
        ),
    },
    # ── Health ─────────────────────────────────────────────────────────────────
    {
        "id": "A021", "category": "Health",
        "title": "AI in Medical Imaging and Diagnosis",
        "tags": ["AI", "medical imaging", "radiology", "diagnosis", "healthcare"],
        "text": (
            "Deep learning models match radiologists in detecting cancers, fractures, and "
            "retinal diseases from imaging data. FDA-cleared AI tools assist in mammography, "
            "colonoscopy, and pathology. Explainability and regulatory approval pathways "
            "shape clinical adoption."
        ),
    },
    {
        "id": "A022", "category": "Health",
        "title": "Mental Health Apps: Technology Meets Therapy",
        "tags": ["mental health", "app", "therapy", "wellbeing", "digital health"],
        "text": (
            "Apps like Calm, Headspace, and Woebot offer CBT exercises, meditation, and "
            "AI-driven mood tracking. Accessibility and affordability make them popular "
            "supplements to traditional therapy. Clinical evidence, privacy, and data "
            "security are scrutinised by regulators."
        ),
    },
    {
        "id": "A023", "category": "Health",
        "title": "Wearable Tech and Preventive Healthcare",
        "tags": ["wearable", "fitness tracker", "smartwatch", "health data", "prevention"],
        "text": (
            "Smartwatches and biosensors monitor heart rate, blood oxygen, sleep, and "
            "glucose in real time. Continuous health data feeds predictive models that "
            "detect early signs of arrhythmia, apnea, and diabetes. Consumer adoption "
            "is accelerating post-pandemic."
        ),
    },
    {
        "id": "A024", "category": "Health",
        "title": "Nutrition Science: Debunking Diet Myths",
        "tags": ["nutrition", "diet", "health", "food science", "wellbeing"],
        "text": (
            "Evidence-based nutrition debunks fad diets and conflicting headlines. "
            "Nutrigenomics personalises dietary advice based on genetics. Ultra-processed "
            "foods and gut microbiome health are leading research frontiers. Food "
            "labelling reforms aim to empower consumer choice."
        ),
    },
    {
        "id": "A025", "category": "Health",
        "title": "Telemedicine: The Future of Healthcare Delivery",
        "tags": ["telemedicine", "remote health", "telehealth", "virtual care", "e-health"],
        "text": (
            "Virtual consultations surged during the pandemic and have become mainstream. "
            "AI triage, remote monitoring devices, and e-prescriptions reduce hospital "
            "burden. Broadband access disparities and reimbursement policies shape "
            "equitable access to telehealth."
        ),
    },
    # ── Travel ─────────────────────────────────────────────────────────────────
    {
        "id": "A026", "category": "Travel",
        "title": "Sustainable Travel: Eco-Friendly Destinations",
        "tags": ["travel", "sustainability", "eco-tourism", "environment", "green travel"],
        "text": (
            "Overtourism damages fragile ecosystems and local communities. Eco-lodges, "
            "carbon offset programmes, and low-impact transport are gaining popularity. "
            "Certifications like Green Key and Rainforest Alliance guide travellers. "
            "Slow travel philosophy encourages deeper cultural immersion."
        ),
    },
    {
        "id": "A027", "category": "Travel",
        "title": "AI-Powered Travel Planning and Personalisation",
        "tags": ["travel", "AI", "personalisation", "booking", "recommendation"],
        "text": (
            "AI chatbots, recommendation engines, and dynamic pricing transform travel "
            "booking. Personalised itineraries based on preferences, budget, and past "
            "trips enhance satisfaction. Airlines and hotels use revenue management AI "
            "to optimise yield."
        ),
    },
    {
        "id": "A028", "category": "Travel",
        "title": "Digital Nomads: Work from Anywhere Culture",
        "tags": ["digital nomad", "remote work", "travel", "lifestyle", "coworking"],
        "text": (
            "Remote work normalisation enables professionals to travel and work "
            "simultaneously. Countries offer digital nomad visas to attract high earners. "
            "Coworking spaces and coliving communities thrive in hotspots like Bali, "
            "Lisbon, and Medellín. Tax and legal complexities require planning."
        ),
    },
    {
        "id": "A029", "category": "Travel",
        "title": "Luxury Travel Trends: Experiential Over Material",
        "tags": ["luxury travel", "experience", "hospitality", "premium", "exclusive"],
        "text": (
            "High-net-worth travellers seek unique experiences — private safaris, "
            "culinary tours, and wellness retreats. Hyper-personalised concierge apps "
            "and AI-powered itinerary curation cater to this segment. Bleisure travel "
            "blends business trips with leisure extensions."
        ),
    },
    {
        "id": "A030", "category": "Travel",
        "title": "Budget Travel Hacks: See the World for Less",
        "tags": ["budget travel", "backpacking", "travel hacks", "affordable", "tips"],
        "text": (
            "Flight deal alerts, flexible date search, and credit-card travel rewards "
            "unlock affordable adventures. Hostels, house-swapping, and Couchsurfing "
            "cut accommodation costs. Off-season travel, street food, and free walking "
            "tours maximise value. Planning and flexibility are key."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
#  2.  USER PROFILES
#      10 simulated website visitors with browsing history & ratings
# ═══════════════════════════════════════════════════════════════════════════════

USERS = [
    {
        "id": "U001",
        "name": "Priya Sharma",
        "age": 28,
        "occupation": "Data Scientist",
        "interests": ["AI", "Technology", "Finance"],
        "browsed": ["A001", "A006", "A007", "A010"],
        "ratings": {"A006": 5, "A007": 4, "A001": 3},
    },
    {
        "id": "U002",
        "name": "Rahul Mehta",
        "age": 35,
        "occupation": "Marketing Manager",
        "interests": ["Marketing", "AI", "Technology"],
        "browsed": ["A016", "A017", "A018", "A019"],
        "ratings": {"A016": 5, "A017": 4, "A018": 3},
    },
    {
        "id": "U003",
        "name": "Ananya Rao",
        "age": 24,
        "occupation": "Medical Student",
        "interests": ["Health", "AI", "Technology"],
        "browsed": ["A021", "A022", "A023", "A024"],
        "ratings": {"A021": 5, "A022": 4, "A023": 5},
    },
    {
        "id": "U004",
        "name": "Karan Patel",
        "age": 30,
        "occupation": "Financial Analyst",
        "interests": ["Finance", "AI", "Technology"],
        "browsed": ["A011", "A012", "A013", "A014"],
        "ratings": {"A011": 5, "A012": 5, "A013": 3},
    },
    {
        "id": "U005",
        "name": "Neha Gupta",
        "age": 26,
        "occupation": "Travel Blogger",
        "interests": ["Travel", "Marketing", "Health"],
        "browsed": ["A026", "A027", "A028", "A029", "A017"],
        "ratings": {"A026": 5, "A027": 4, "A028": 5, "A029": 3},
    },
    {
        "id": "U006",
        "name": "Aditya Singh",
        "age": 32,
        "occupation": "Software Engineer",
        "interests": ["Technology", "AI", "Finance"],
        "browsed": ["A002", "A003", "A004", "A006", "A012"],
        "ratings": {"A003": 5, "A004": 4, "A006": 5},
    },
    {
        "id": "U007",
        "name": "Sanya Kapoor",
        "age": 22,
        "occupation": "College Student",
        "interests": ["AI", "Marketing", "Travel"],
        "browsed": ["A006", "A009", "A017", "A028", "A030"],
        "ratings": {"A006": 4, "A009": 5, "A028": 4},
    },
    {
        "id": "U008",
        "name": "Vikram Nair",
        "age": 45,
        "occupation": "CEO",
        "interests": ["Finance", "Technology", "Marketing"],
        "browsed": ["A014", "A015", "A003", "A016", "A012"],
        "ratings": {"A014": 5, "A015": 4, "A016": 5},
    },
    {
        "id": "U009",
        "name": "Meera Iyer",
        "age": 38,
        "occupation": "Nutritionist",
        "interests": ["Health", "Travel", "Marketing"],
        "browsed": ["A023", "A024", "A025", "A026", "A022"],
        "ratings": {"A024": 5, "A025": 4, "A023": 3},
    },
    {
        "id": "U010",
        "name": "Dev Bose",
        "age": 29,
        "occupation": "Entrepreneur",
        "interests": ["Finance", "AI", "Marketing"],
        "browsed": ["A011", "A006", "A009", "A016", "A015"],
        "ratings": {"A006": 4, "A011": 5, "A016": 4},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ContentRecommender:
    """
    Hybrid Recommendation Engine
    ─────────────────────────────
    Content-Based  : TF-IDF on article text + tags → cosine similarity
                     to the union of articles a user has browsed/rated.
    Collaborative  : User-item implicit matrix → cosine similarity between
                     users → weighted average of neighbour ratings.
    Hybrid Score   : α × content_score + (1-α) × collab_score
    """

    ALPHA = 0.60            # weight for content-based component
    TOP_N = 5               # default number of recommendations
    MIN_COLLAB_USERS = 2    # minimum similar users needed for collab signal

    def __init__(self, articles: list, users: list):
        self.articles = articles
        self.users = users
        self._article_index  = {a["id"]: i for i, a in enumerate(articles)}
        self._user_index     = {u["id"]: i for i, u in enumerate(users)}

        # ── TF-IDF matrix  [n_articles × n_features] ──────────────────────────
        corpus = [
            a["text"] + " " + " ".join(a["tags"]) + " " + a["category"]
            for a in articles
        ]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)   # sparse

        # ── User-item implicit feedback matrix  [n_users × n_articles] ────────
        n_users    = len(users)
        n_articles = len(articles)
        self._ui_matrix = np.zeros((n_users, n_articles), dtype=np.float32)

        for user in users:
            ui = self._user_index[user["id"]]
            for art_id in user.get("browsed", []):
                if art_id in self._article_index:
                    ai = self._article_index[art_id]
                    self._ui_matrix[ui, ai] += 1.0          # implicit feedback
            for art_id, rating in user.get("ratings", {}).items():
                if art_id in self._article_index:
                    ai = self._article_index[art_id]
                    self._ui_matrix[ui, ai] = max(
                        self._ui_matrix[ui, ai], rating / 5.0
                    )                                         # normalise to [0,1]

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _content_scores(self, user: dict) -> np.ndarray:
        """Return a content-based score vector [n_articles]."""
        interacted = set(user.get("browsed", [])) | set(user.get("ratings", {}).keys())
        if not interacted:
            return np.zeros(len(self.articles))

        indices = [self._article_index[a] for a in interacted if a in self._article_index]
        # .mean(axis=0) on a sparse matrix returns numpy.matrix; convert to array
        user_profile = np.asarray(self._tfidf_matrix[indices].mean(axis=0))  # (1, n_features)
        scores = cosine_similarity(user_profile, self._tfidf_matrix).flatten()
        return scores

    def _collab_scores(self, user: dict) -> np.ndarray:
        """Return a collaborative-filtering score vector [n_articles]."""
        ui = self._user_index[user["id"]]
        user_vec = self._ui_matrix[ui].reshape(1, -1)

        # similarity to every other user
        sims = cosine_similarity(user_vec, self._ui_matrix).flatten()
        sims[ui] = 0.0      # exclude self

        # weighted sum of other users' interactions
        if sims.max() == 0:
            return np.zeros(len(self.articles))

        weighted = sims.dot(self._ui_matrix)         # [n_articles]
        normaliser = sims.sum() + 1e-9
        return weighted / normaliser

    def _hybrid_scores(self, user: dict) -> np.ndarray:
        """Blend content and collab scores."""
        cs = self._content_scores(user)
        co = self._collab_scores(user)
        return self.ALPHA * cs + (1 - self.ALPHA) * co

    def _explain(self, user: dict, art_id: str) -> str:
        """Generate a human-readable explanation for why an article is recommended."""
        article = self.articles[self._article_index[art_id]]
        interacted = set(user.get("browsed", [])) | set(user.get("ratings", {}).keys())

        # Find the most similar already-read article
        if interacted:
            interacted_titles = {
                a["id"]: a["title"]
                for a in self.articles if a["id"] in interacted
            }
            indices      = [self._article_index[a] for a in interacted if a in self._article_index]
            rec_idx      = self._article_index[art_id]
            rec_vec      = self._tfidf_matrix[rec_idx]
            sims         = cosine_similarity(rec_vec, self._tfidf_matrix[indices]).flatten()
            best_local   = int(np.argmax(sims))
            best_id      = list(interacted_titles.keys())[best_local]
            best_title   = interacted_titles[best_id]
            return f"Because you read \"{best_title}\" · Category: {article['category']}"

        return f"Matches your interest in {', '.join(user.get('interests', ['this topic']))}"

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────────

    def recommend(self, user_id: str, n: int = None, exclude_seen: bool = True) -> list:
        """
        Return top-n recommendations for *user_id*.

        Returns
        -------
        list of dicts:
          { article_id, title, category, score, explanation }
        """
        if n is None:
            n = self.TOP_N

        user = next((u for u in self.users if u["id"] == user_id), None)
        if user is None:
            return []

        scores = self._hybrid_scores(user)

        interacted = set(user.get("browsed", [])) | set(user.get("ratings", {}).keys())
        results = []
        for i, article in enumerate(self.articles):
            if exclude_seen and article["id"] in interacted:
                continue
            results.append((scores[i], article))

        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:n]

        output = []
        for score, article in results:
            output.append({
                "article_id":  article["id"],
                "title":       article["title"],
                "category":    article["category"],
                "score":       round(float(score), 4),
                "explanation": self._explain(user, article["id"]),
            })
        return output

    def add_rating(self, user_id: str, article_id: str, rating: int) -> None:
        """Record a new rating (1-5) and update the UI matrix on-the-fly."""
        user = next((u for u in self.users if u["id"] == user_id), None)
        if user is None or article_id not in self._article_index:
            return
        user.setdefault("ratings", {})[article_id] = rating
        if article_id not in user.get("browsed", []):
            user.setdefault("browsed", []).append(article_id)
        ui = self._user_index[user_id]
        ai = self._article_index[article_id]
        self._ui_matrix[ui, ai] = rating / 5.0

    def evaluate(self, k: int = 5) -> dict:
        """
        Leave-one-out evaluation (hit-rate@k).
        Hides the last rated article for each user, generates recs, checks if
        the hidden article appears in top-k.
        """
        hits = 0
        evaluated = 0
        for user in self.users:
            ratings = user.get("ratings", {})
            if len(ratings) < 2:
                continue
            # hold out one rated article
            hold_out_id = list(ratings.keys())[-1]
            # create a temporary copy
            temp_user = copy.deepcopy(user)
            temp_user["ratings"].pop(hold_out_id, None)
            if hold_out_id in temp_user.get("browsed", []):
                temp_user["browsed"].remove(hold_out_id)

            # temporarily swap in the modified user
            idx = self._user_index[user["id"]]
            orig_row = self._ui_matrix[idx].copy()
            self._ui_matrix[idx, self._article_index[hold_out_id]] = 0.0

            recs = self.recommend(user["id"], n=k, exclude_seen=True)
            rec_ids = [r["article_id"] for r in recs]

            if hold_out_id in rec_ids:
                hits += 1
            evaluated += 1

            # restore
            self._ui_matrix[idx] = orig_row

        hit_rate = hits / evaluated if evaluated > 0 else 0.0
        return {"evaluated_users": evaluated, "hits": hits, "hit_rate_at_k": round(hit_rate, 3), "k": k}


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_COLORS = {
    "Technology": "\033[94m",   # Blue
    "AI":         "\033[95m",   # Magenta
    "Finance":    "\033[92m",   # Green
    "Marketing":  "\033[93m",   # Yellow
    "Health":     "\033[91m",   # Red
    "Travel":     "\033[96m",   # Cyan
}
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

DIVIDER = "─" * 70


def cat_color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "")


def print_header(title: str) -> None:
    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")


def print_recommendations(user: dict, recs: list) -> None:
    print(f"\n{BOLD}  Top Recommendations for {user['name']} "
          f"({user['occupation']}){RESET}")
    print(f"  Interests: {', '.join(user['interests'])}")
    print(f"  {DIM}Articles read: {len(user.get('browsed', []))}{RESET}\n")
    print(f"  {'#':<4} {'ID':<6} {'Score':<8} {'Category':<12} {'Title':<35} Explanation")
    print(f"  {DIVIDER}")
    for i, rec in enumerate(recs, 1):
        cat  = rec["category"]
        col  = cat_color(cat)
        title = rec["title"][:33] + ".." if len(rec["title"]) > 35 else rec["title"]
        expl  = rec["explanation"][:45] + ".." if len(rec["explanation"]) > 47 else rec["explanation"]
        print(f"  {i:<4} {rec['article_id']:<6} {rec['score']:<8.4f} "
              f"{col}{cat:<12}{RESET} {title:<35} {DIM}{expl}{RESET}")


def print_article_detail(art: dict) -> None:
    col = cat_color(art["category"])
    print(f"\n  {BOLD}{art['title']}{RESET}  "
          f"[{col}{art['category']}{RESET}]  {DIM}ID: {art['id']}{RESET}")
    print(f"  Tags: {', '.join(art['tags'])}")
    wrapped = textwrap.fill(art["text"], width=66, initial_indent="  ",
                            subsequent_indent="  ")
    print(f"\n{wrapped}")


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  AUTOMATED SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_self_test(engine: ContentRecommender) -> None:
    print_header("SELF-TEST: Recommendations for All Users")

    for user in engine.users:
        recs = engine.recommend(user["id"], n=5)
        rec_ids = [r["article_id"] for r in recs]
        # Assertions
        assert all(rid in engine._article_index for rid in rec_ids), \
            f"Invalid article ID returned for {user['id']}"
        assert len(recs) <= 5, "Too many recommendations returned"
        print_recommendations(user, recs)
        print()

    # Hit-rate evaluation
    print_header("EVALUATION: Hit-Rate @ 5 (Leave-One-Out)")
    metrics = engine.evaluate(k=5)
    print(f"\n  Evaluated Users : {metrics['evaluated_users']}")
    print(f"  Hits (k={metrics['k']})     : {metrics['hits']}")
    print(f"  Hit-Rate @ {metrics['k']}   : {metrics['hit_rate_at_k']:.1%}")
    print(f"\n  {DIM}Interpretation: % of users for whom the held-out article "
          f"appears in Top-{metrics['k']} recs{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  INTERACTIVE CLI DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_menu(engine: ContentRecommender) -> None:
    """Menu-driven session to explore the recommendation system."""
    while True:
        print_header("AI CONTENT RECOMMENDATION SYSTEM — Main Menu")
        print("  1.  List all website users")
        print("  2.  Get personalized recommendations for a user")
        print("  3.  Browse the content catalog")
        print("  4.  Rate an article (and refresh recommendations)")
        print("  5.  Run evaluation metrics")
        print("  6.  Exit")
        print()
        choice = input("  Enter choice (1-6): ").strip()

        # ── 1. List users ──────────────────────────────────────────────────────
        if choice == "1":
            print_header("WEBSITE USERS")
            print(f"\n  {'ID':<6} {'Name':<18} {'Age':<5} {'Occupation':<22} Interests")
            print(f"  {DIVIDER}")
            for u in engine.users:
                print(f"  {u['id']:<6} {u['name']:<18} {u['age']:<5} "
                      f"{u['occupation']:<22} {', '.join(u['interests'])}")

        # ── 2. Recommendations ────────────────────────────────────────────────
        elif choice == "2":
            uid = input("  Enter User ID (e.g. U001): ").strip().upper()
            user = next((u for u in engine.users if u["id"] == uid), None)
            if user is None:
                print(f"  [!] User '{uid}' not found.")
                continue
            recs = engine.recommend(uid, n=5)
            if not recs:
                print("  [!] No recommendations available (user has seen everything!).")
            else:
                print_recommendations(user, recs)

        # ── 3. Catalog ────────────────────────────────────────────────────────
        elif choice == "3":
            print_header("CONTENT CATALOG")
            cats = sorted(set(a["category"] for a in ARTICLES))
            print(f"  Categories: {', '.join(cats)}\n")
            filt = input(
                "  Filter by category (or press Enter to list all): "
            ).strip().title()
            filtered = ARTICLES if not filt else [a for a in ARTICLES if a["category"] == filt]
            if not filtered:
                print(f"  [!] No articles in category '{filt}'.")
                continue
            print(f"\n  {'ID':<6} {'Category':<12} Title")
            print(f"  {DIVIDER}")
            for a in filtered:
                col = cat_color(a["category"])
                print(f"  {a['id']:<6} {col}{a['category']:<12}{RESET} {a['title']}")
            detail = input("\n  Enter Article ID to read full detail (or Enter to skip): ").strip().upper()
            if detail:
                art = next((a for a in ARTICLES if a["id"] == detail), None)
                if art:
                    print_article_detail(art)
                else:
                    print("  [!] Article not found.")

        # ── 4. Rate article ───────────────────────────────────────────────────
        elif choice == "4":
            uid = input("  Enter User ID: ").strip().upper()
            user = next((u for u in engine.users if u["id"] == uid), None)
            if user is None:
                print(f"  [!] User '{uid}' not found.")
                continue
            art_id = input("  Enter Article ID to rate: ").strip().upper()
            art = next((a for a in ARTICLES if a["id"] == art_id), None)
            if art is None:
                print(f"  [!] Article '{art_id}' not found.")
                continue
            try:
                rating = int(input(f"  Your rating for \"{art['title']}\" (1-5): ").strip())
                if not 1 <= rating <= 5:
                    raise ValueError
            except ValueError:
                print("  [!] Invalid rating. Enter a number from 1 to 5.")
                continue
            engine.add_rating(uid, art_id, rating)
            print(f"\n  ✓  Rating saved!  Refreshing recommendations for {user['name']}…")
            recs = engine.recommend(uid, n=5)
            print_recommendations(user, recs)

        # ── 5. Evaluation ─────────────────────────────────────────────────────
        elif choice == "5":
            print_header("EVALUATION METRICS")
            metrics = engine.evaluate(k=5)
            print(f"\n  Leave-One-Out  Hit-Rate @ 5")
            print(f"  Evaluated users : {metrics['evaluated_users']}")
            print(f"  Hits            : {metrics['hits']}")
            print(f"  Hit-Rate        : {metrics['hit_rate_at_k']:.1%}")

        # ── 6. Exit ───────────────────────────────────────────────────────────
        elif choice == "6":
            print(f"\n  {BOLD}Goodbye!{RESET}  Thank you for exploring the "
                  f"AI Content Recommendation System.\n")
            break

        else:
            print("  [!] Invalid choice. Please enter a number from 1 to 6.")

        input(f"\n  {DIM}Press Enter to return to the menu…{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{BOLD}Initialising Recommendation Engine…{RESET}")
    engine = ContentRecommender(ARTICLES, USERS)
    print("  ✓  TF-IDF matrix built  "
          f"({len(ARTICLES)} articles × {engine._tfidf_matrix.shape[1]} features)")
    print(f"  ✓  User-Item matrix built  ({len(USERS)} users × {len(ARTICLES)} articles)")

    # Automated self-test (runs once at startup)
    run_self_test(engine)

    # Drop into interactive session
    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}  Launching Interactive Demo…{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")
    interactive_menu(engine)


if __name__ == "__main__":
    # Pass --test to skip the interactive menu (useful for automated checks)
    if "--test" in sys.argv:
        print("\nInitialising Recommendation Engine\u2026")
        engine = ContentRecommender(ARTICLES, USERS)
        print(f"  \u2713  TF-IDF matrix built  "
              f"({len(ARTICLES)} articles \u00d7 {engine._tfidf_matrix.shape[1]} features)")
        print(f"  \u2713  User-Item matrix built  ({len(USERS)} users \u00d7 {len(ARTICLES)} articles)")
        run_self_test(engine)
    else:
        main()
