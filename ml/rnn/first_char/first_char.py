import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, Counter
import re
import random

class CharToWordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2):
        super(CharToWordRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer for input characters
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Output layer to predict word indices
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Get the last output for each sequence
        output = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return output, hidden

class FirstCharPredictor:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.model = None
        
    def build_vocabulary(self, sentences):
        """Build vocabulary from training sentences"""
        # Collect all unique first characters and words
        chars = set()
        words = set()
        
        for sentence in sentences:
            words_in_sentence = sentence.lower().split()
            for word in words_in_sentence:
                # Clean word (remove punctuation)
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    chars.add(clean_word[0])
                    words.add(clean_word)
        
        # Add special tokens
        chars.add('<PAD>')
        chars.add('<UNK>')
        words.add('<PAD>')
        words.add('<UNK>')
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(words))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Built vocabulary: {len(self.char_to_idx)} characters, {len(self.word_to_idx)} words")
        
    def prepare_data(self, sentences, max_length=10):
        """Convert sentences to training data"""
        X, y = [], []
        
        for sentence in sentences:
            words = sentence.lower().split()
            if len(words) > max_length:
                continue
                
            # Extract first characters and clean words
            chars = []
            clean_words = []
            
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    chars.append(clean_word[0])
                    clean_words.append(clean_word)
            
            if len(chars) > 1:  # Need at least 2 words for training
                # Convert to indices
                char_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in chars]
                word_indices = [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in clean_words]
                
                # Pad sequences
                while len(char_indices) < max_length:
                    char_indices.append(self.char_to_idx['<PAD>'])
                    word_indices.append(self.word_to_idx['<PAD>'])
                
                X.append(char_indices[:max_length])
                y.append(word_indices[:max_length])
        
        return torch.tensor(X), torch.tensor(y)
    
    def train(self, sentences, epochs=100, lr=0.001):
        """Train the RNN model"""
        self.build_vocabulary(sentences)
        X, y = self.prepare_data(sentences)
        
        vocab_size = len(self.word_to_idx)
        self.model = CharToWordRNN(vocab_size)
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.word_to_idx['<PAD>'])
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.model(X)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, first_chars):
        """Predict words from first characters"""
        if self.model is None:
            return "Model not trained yet!"
        
        self.model.eval()
        with torch.no_grad():
            # Convert input to indices
            char_indices = []
            for char in first_chars.lower().replace(' ', ''):
                char_indices.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
            
            # Pad to model's expected length
            max_len = 10
            while len(char_indices) < max_len:
                char_indices.append(self.char_to_idx['<PAD>'])
            
            # Make prediction
            input_tensor = torch.tensor([char_indices[:max_len]])
            output, _ = self.model(input_tensor)
            
            # Get predicted words
            predicted_indices = torch.argmax(output[0], dim=1)
            predicted_words = []
            
            for idx in predicted_indices:
                word = self.idx_to_word[idx.item()]
                if word != '<PAD>':
                    predicted_words.append(word)
                else:
                    break
            
            return ' '.join(predicted_words[:len(first_chars.replace(' ', ''))])

# Example usage and training
def main():
    # Sample training data - you can expand this with more sentences
    training_sentences = [
        "thank god it is friday",
        "happy new year everyone",
        "good morning my friend",
        "see you later alligator",
        "break a leg tonight",
        "time to go home",
        "have a great day",
        "nice to meet you",
        "call me when ready",
        "just do it now",
        "make it happen today",
        "keep up the work",
        "never give up hope",
        "always believe in yourself",
        "work hard play harder",
        "live life to fullest",
        "dream big think positive",
        "learn something new daily",
        "practice makes perfect sense",
        "knowledge is real power",
        "time flies when having fun",
        "money cannot buy true happiness",
        "health is more important",
        "family comes first always",
        "friends are life treasures",
        "love makes world beautiful",
        "music soothes the soul",
        "books open new worlds",
        "travel broadens the mind",
        "exercise keeps body healthy",
        "sleep is very important",
        "water is life essential",
        "food gives us energy",
        "sun provides natural light",
        "rain helps plants grow",
        "flowers smell so sweet",
        "birds sing beautiful songs",
        "cats are cute pets",
        "dogs are loyal companions",
        "trees give us oxygen",
        "oceans are vast deep"
    ]
    
    # Create and train the predictor
    predictor = FirstCharPredictor()
    print("Training the model...")
    predictor.train(training_sentences, epochs=150)
    
    # Test predictions
    test_cases = [
        "t g i f",      # thank god it friday
        "h n y",        # happy new year
        "g m m f",      # good morning my friend
        "s y l",        # see you later
        "b a l",        # break a leg
        "t t g h",      # time to go home
        "h a g d",      # have a great day
    ]
    
    print("\n" + "="*50)
    print("PREDICTIONS:")
    print("="*50)
    
    for test in test_cases:
        prediction = predictor.predict(test)
        print(f"Input: '{test}' -> Predicted: '{prediction}'")

if __name__ == "__main__":
    main()