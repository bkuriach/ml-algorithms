import numpy as np
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class TFIDFVectorizer:
    def __init__(self, max_features=None, min_df=1, max_df=1.0, stop_words=None):
        """
        TF-IDF Vectorizer implementation from scratch
        
        Args:
            max_features: Maximum number of features to use
            min_df: Minimum document frequency (ignore terms that appear in fewer docs)
            max_df: Maximum document frequency (ignore terms that appear in more docs)
            stop_words: List of stop words to ignore
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words if stop_words else []
        
        self.vocabulary_ = {}
        self.idf_ = {}
        self.feature_names_ = []
    
    def _preprocess_text(self, text):
        """Clean and tokenize text"""
        # Convert to lowercase and remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Split into words
        words = text.split()
        # Remove stop words
        words = [word for word in words if word not in self.stop_words]
        return words
    
    def _build_vocabulary(self, documents):
        """Build vocabulary from documents"""
        # Count document frequency for each term
        doc_freq = Counter()
        total_docs = len(documents)
        
        for doc in documents:
            words = self._preprocess_text(doc)
            unique_words = set(words)
            for word in unique_words:
                doc_freq[word] += 1
        
        # Filter terms based on min_df and max_df
        filtered_vocab = {}
        for word, freq in doc_freq.items():
            if freq >= self.min_df and freq <= (self.max_df * total_docs):
                filtered_vocab[word] = freq
        
        # Limit vocabulary size if max_features is specified
        if self.max_features and len(filtered_vocab) > self.max_features:
            # Keep most frequent terms
            sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
            filtered_vocab = dict(sorted_vocab[:self.max_features])
        
        # Create vocabulary mapping
        self.vocabulary_ = {word: idx for idx, word in enumerate(filtered_vocab.keys())}
        self.feature_names_ = list(self.vocabulary_.keys())
        
        # Calculate IDF values
        self._calculate_idf(filtered_vocab, total_docs)
    
    def _calculate_idf(self, doc_freq, total_docs):
        """Calculate inverse document frequency"""
        self.idf_ = {}
        for word in self.vocabulary_:
            # IDF = log(N / df) where N is total docs and df is document frequency
            self.idf_[word] = np.log(total_docs / doc_freq[word])
    
    def _calculate_tf(self, words):
        """Calculate term frequency for a document"""
        word_count = len(words)
        tf = Counter(words)
        # Normalize by total word count
        for word in tf:
            tf[word] = tf[word] / word_count
        return tf
    
    def fit(self, documents):
        """Fit the vectorizer on documents"""
        self._build_vocabulary(documents)
        return self
    
    def transform(self, documents):
        """Transform documents to TF-IDF matrix"""
        if not self.vocabulary_:
            raise ValueError("Vectorizer must be fitted before transforming")
        
        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary_)))
        
        for doc_idx, doc in enumerate(documents):
            words = self._preprocess_text(doc)
            tf = self._calculate_tf(words)
            
            for word, tf_value in tf.items():
                if word in self.vocabulary_:
                    word_idx = self.vocabulary_[word]
                    idf_value = self.idf_[word]
                    tfidf_matrix[doc_idx, word_idx] = tf_value * idf_value
        
        return tfidf_matrix
    
    def fit_transform(self, documents):
        """Fit and transform documents"""
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self):
        """Get feature names (vocabulary)"""
        return self.feature_names_


# Binary Text Classification Example
def create_sample_data():
    """Create sample movie review data for binary classification"""
    positive_reviews = [
        "This movie is absolutely fantastic and amazing",
        "I loved every moment of this brilliant film",
        "Outstanding performance and excellent storyline",
        "Great acting and wonderful cinematography",
        "Best movie I have ever seen, highly recommended",
        "Incredible plot and superb direction",
        "Amazing characters and beautiful scenes",
        "Perfect entertainment and great fun to watch",
        "Excellent movie with outstanding performances",
        "Wonderful story and fantastic acting throughout"
    ]
    
    negative_reviews = [
        "This movie is terrible and boring",
        "Worst film I have ever watched, complete waste of time",
        "Poor acting and awful storyline",
        "Bad direction and horrible performances",
        "Terrible movie, very disappointing and boring",
        "Awful plot and poor character development",
        "Bad cinematography and terrible acting",
        "Boring movie with no interesting moments",
        "Poor storyline and disappointing performances",
        "Terrible direction and awful script writing"
    ]
    
    # Combine data
    documents = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1 = positive, 0 = negative
    
    return documents, labels


if __name__ == "__main__":
    # Create sample data
    documents, labels = create_sample_data()
    
    print(f"Total documents: {len(documents)}")
    print(f"Positive reviews: {sum(labels)}")
    print(f"Negative reviews: {len(labels) - sum(labels)}")
    
    # Define common stop words
    stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should']
    
    # Initialize TF-IDF vectorizer
    tfidf = TFIDFVectorizer(
        max_features=100,
        min_df=1,
        max_df=0.8,
        stop_words=stop_words
    )
    
    # Transform documents to TF-IDF matrix
    X = tfidf.fit_transform(documents)
    y = np.array(labels)
    
    print(f"\nTF-IDF Matrix shape: {X.shape}")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Logistic Regression
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Show feature importance
    feature_names = tfidf.get_feature_names()
    coefficients = classifier.coef_[0]
    
    # Get top positive and negative features
    feature_importance = list(zip(feature_names, coefficients))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Positive Features (words that indicate positive sentiment):")
    for word, coef in feature_importance[:10]:
        print(f"{word}: {coef:.3f}")
    
    print("\nTop 10 Negative Features (words that indicate negative sentiment):")
    for word, coef in feature_importance[-10:]:
        print(f"{word}: {coef:.3f}")
    
    # Test on new samples
    print("\n" + "="*50)
    print("Testing on new samples:")
    
    new_reviews = [
        "This movie is amazing and I loved it",
        "Terrible film, very boring and disappointing",
        "Great storyline with excellent acting"
    ]
    
    X_new = tfidf.transform(new_reviews)
    predictions = classifier.predict(X_new)
    probabilities = classifier.predict_proba(X_new)
    
    for i, review in enumerate(new_reviews):
        sentiment = "Positive" if predictions[i] == 1 else "Negative"
        confidence = max(probabilities[i]) * 100
        print(f"\nReview: '{review}'")
        print(f"Prediction: {sentiment} (Confidence: {confidence:.1f}%)")