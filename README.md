# import re
from collections import defaultdict, deque
from math import log, exp
import random

class ContextualAutocomplete:
    def __init__(self, decay_factor=0.9, context_weight=0.3):
        """
        decay_factor: Controls how quickly context importance fades (0.9 = retains 90% importance per interaction)
        context_weight: Balances between global frequency and contextual relevance (0.3 = 30% contextual influence)
        """
        self.trie = {}  # Prefix tree for autocomplete
        self.context_history = defaultdict(lambda: deque(maxlen=10))  # User context history
        self.global_freq = defaultdict(int)  # Global frequency of terms
        self.context_freq = defaultdict(lambda: defaultdict(float))  # Context-specific frequencies
        self.decay_factor = decay_factor
        self.context_weight = context_weight

    def index_term(self, term, category=None):
        """Add a term to the autocomplete index with optional category"""
        term = term.lower().strip()
        if not term:
            return
            
        # Update trie structure
        node = self.trie
        for char in term:
            node = node.setdefault(char, {})
        node['$'] = term  # Mark end of term
        
        # Update global frequency
        self.global_freq[term] += 1
        
        # Add to category context if provided
        if category:
            self.context_history[category].append(term)
            self._update_context_freq(category, term)

    def _update_context_freq(self, category, term):
        """Update context-specific frequency with decay"""
        # Apply decay to all terms in this context
        for t in list(self.context_freq[category].keys()):
            self.context_freq[category][t] *= self.decay_factor
            
        # Add new occurrence
        self.context_freq[category][term] += 1.0

    def _get_context_bias(self, category, term):
        """Calculate contextual bias score for a term in a category"""
        if not category or term not in self.context_freq[category]:
            return 1.0  # Neutral bias if no context
        
        # Context score = (context_freq / max_context_freq) * weight
        max_freq = max(self.context_freq[category].values()) if self.context_freq[category] else 1.0
        context_score = (self.context_freq[category][term] / max_freq) * self.context_weight
        
        # Combine with global popularity to avoid obscurity bias
        global_score = log(self.global_freq.get(term, 1)) / log(max(self.global_freq.values() or [1]))
        
        return context_score + (1.0 - self.context_weight) * global_score

    def search(self, prefix, category=None, limit=5):
        """Get autocomplete suggestions with contextual biasing"""
        prefix = prefix.lower().strip()
        if not prefix or prefix[0] not in self.trie:
            return []
            
        # Traverse trie to find prefix node
        node = self.trie
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
            
        # Collect all terms starting with prefix
        suggestions = []
        self._collect_terms(node, suggestions)
        
        # Score and sort suggestions
        scored_suggestions = []
        for term in suggestions:
            # Base score: logarithmic frequency
            base_score = log(self.global_freq.get(term, 1) + 1)
            
            # Apply contextual bias
            context_bias = self._get_context_bias(category, term)
            
            # Final score = base * bias
            final_score = base_score * context_bias
            
            scored_suggestions.append((final_score, term))
        
        # Sort by score descending, then alphabetically
        scored_suggestions.sort(key=lambda x: (-x[0], x[1]))
        
        return [term for _, term in scored_suggestions[:limit]]

    def _collect_terms(self, node, results):
        """Recursively collect all terms from a trie node"""
        if '$' in node:
            results.append(node['$'])
            
        for char, child in node.items():
            if char != '$':
                self._collect_terms(child, results)

    def simulate_user_interaction(self, category, search_term):
        """Simulate a user search interaction to build context"""
        self.index_term(search_term, category)
        # Optionally return autocomplete results for this interaction
        return self.search(search_term, category)

# Example usage
if __name__ == "__main__":
    autocomplete = ContextualAutocomplete(decay_factor=0.85, context_weight=0.4)
    
    # Index sample terms with categories
    autocomplete.index_term("apple", "fruits")
    autocomplete.index_term("apricot", "fruits")
    autocomplete.index_term("avocado", "fruits")
    autocomplete.index_term("asparagus", "vegetables")
    autocomplete.index_term("artichoke", "vegetables")
    autocomplete.index_term("alarm clock", "electronics")
    autocomplete.index_term("apple watch", "electronics")
    
    # Simulate user interactions in "fruits" category
    autocomplete.simulate_user_interaction("fruits", "apple")
    autocomplete.simulate_user_interaction("fruits", "avocado")
    
    # Test autocomplete with context
    print("Suggestions for 'a' in fruits:", autocomplete.search("a", "fruits"))
    print("Suggestions for 'a' in electronics:", autocomplete.search("a", "electronics"))
    print("Suggestions for 'a' (no context):", autocomplete.search("a"))
    
    # Simulate new interaction to update context
    autocomplete.simulate_user_interaction("fruits", "apricot")
    print("Updated suggestions for 'a' in fruits:", autocomplete.search("a", "fruits"))
