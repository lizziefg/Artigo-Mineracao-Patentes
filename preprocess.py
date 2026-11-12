"""
Módulo de Pré-processamento Especializado de Patentes (Sprint E2)
Remoção de ruído formal de redação patentária (patentese), normalização e tokenização.
"""

import re
import string
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import PATENTESE_STOPWORDS

# Download silencioso de recursos necessários do NLTK
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

class PatentTextPreprocessor:
    """
    Pré-processador de texto especializado para patentes tecnológicas (CPC G06N).
    Elimina jargões legais/redacionais ('patentese') e preserva termos técnicos de IA.
    """
    def __init__(self, custom_stopwords: set = None, min_word_len: int = 3):
        self.min_word_len = min_word_len
        
        # Stopwords padrão em inglês
        try:
            self.standard_stopwords = set(stopwords.words('english'))
        except Exception:
            self.standard_stopwords = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
                "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
                "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that"
            }
        
        # Consolidação das stopwords do patentese
        self.patentese_stopwords = set(s.lower() for s in (custom_stopwords or PATENTESE_STOPWORDS))
        self.all_stopwords = self.standard_stopwords.union(self.patentese_stopwords)
        
        # Lematizador
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    @staticmethod
    def is_english_text(text: str, min_len: int = 40) -> bool:
        """Verifica se o texto é predominantemente em inglês (evita ruído multilíngue no corpus)."""
        if not isinstance(text, str) or len(text) < min_len:
            return False
        ascii_letters = sum(1 for c in text if "a" <= c.lower() <= "z")
        if ascii_letters / max(len(text), 1) < 0.65:
            return False
        low = f" {text.lower()} "
        en_hits = sum(
            1 for w in (" the ", " and ", " of ", " to ", " in ", " a ", " is ", " for ", " with ")
            if w in low
        )
        non_en_hits = sum(
            1 for w in (" der ", " die ", " und ", " von ", " zu ", " le ", " la ", " des ", " les ", " une ")
            if w in low
        )
        return en_hits >= 2 and en_hits > non_en_hits

    def clean_raw_text(self, text: str) -> str:
        """Limpa ruídos estruturais, referências numéricas de patentes e caracteres especiais."""
        if not isinstance(text, str):
            return ""
        
        # Decodificar caracteres comuns
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        
        # Remover referências a números de reivindicação / figuras ex: [0012], (FIG. 1), (102)
        text = re.sub(r'\[\s*\d+\s*\]', ' ', text)
        text = re.sub(r'\(\s*(?:fig|figure|see)\s*\.?\s*\d+[a-z]?\s*\)', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\(\s*\d+\s*\)', ' ', text)
        
        # Remover números e códigos de patentes avulsos
        text = re.sub(r'\bUS\s*\d+[A-Z\d]*\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bEP\s*\d+[A-Z\d]*\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bWO\s*\d+[A-Z\d]*\b', ' ', text, flags=re.IGNORECASE)
        
        # Remover URLs e emails
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Substituir hífens entre palavras por espaço ou manter compostos conhecidos
        text = re.sub(r'(?<=\w)-(?=\w)', ' ', text)
        
        # Manter apenas letras e espaços
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Colapsar múltiplos espaços
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def tokenize_and_filter(self, text: str) -> List[str]:
        """Tokeniza, remove patentese stopwords e lematiza termos preservando vocabulário de IA."""
        clean_text = self.clean_raw_text(text)
        words = clean_text.split()
        
        filtered_tokens = []
        for word in words:
            if len(word) < self.min_word_len:
                continue
            if word in self.all_stopwords:
                continue
            
            # Lematização se disponível
            if self.lemmatizer:
                try:
                    word = self.lemmatizer.lemmatize(word)
                except Exception:
                    pass
            
            # Revalidação pós-lematização
            if word not in self.all_stopwords and len(word) >= self.min_word_len:
                filtered_tokens.append(word)
                
        return filtered_tokens

    def process_for_models(self, raw_text: str) -> Tuple[str, List[str]]:
        """
        Retorna tanto a string limpa (para BERTopic, que prefere texto contínuo contextual)
        quanto a lista de tokens (para LDA e cálculo de coerência).
        """
        tokens = self.tokenize_and_filter(raw_text)
        cleaned_sentence = " ".join(tokens)
        return cleaned_sentence, tokens
