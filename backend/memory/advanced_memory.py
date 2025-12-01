"""Advanced memory system with vector-based similarity search"""
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using basic memory")

class AdvancedEpisodeMemory:
    """Advanced memory system with vector-based similarity search"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episodes = []  # Fallback in-memory storage
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_embeddings = True
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.use_embeddings = False
        else:
            self.use_embeddings = False
    
    def store_episode(self, episode_data: Dict[str, Any]):
        """Store episode in memory with semantic indexing"""
        
        # Add timestamp and ID
        episode_data['timestamp'] = episode_data.get('timestamp', datetime.utcnow().isoformat())
        episode_data['episode_id'] = self._generate_episode_id(episode_data)
        
        # Create embedding if available
        if self.use_embeddings:
            memory_text = f"{episode_data['instruction']} {episode_data.get('result', '')}"
            try:
                episode_data['embedding'] = self.encoder.encode(memory_text).tolist()
            except Exception as e:
                logger.error(f"Failed to create embedding: {e}")
                episode_data['embedding'] = None
        
        self.episodes.append(episode_data)
        logger.info(f"Stored episode {episode_data['episode_id']} for instruction: {episode_data['instruction']}")
    
    def retrieve_similar_episodes(self, instruction: str, app_context: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar past episodes using semantic search"""
        
        if not self.episodes:
            return []
        
        if self.use_embeddings:
            try:
                query_embedding = self.encoder.encode(instruction)
                
                # Calculate cosine similarity
                similarities = []
                for episode in self.episodes:
                    if episode.get('embedding'):
                        ep_embedding = np.array(episode['embedding'])
                        similarity = np.dot(query_embedding, ep_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(ep_embedding)
                        )
                        similarities.append((similarity, episode))
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x[0], reverse=True)
                return [
                    {
                        'score': float(score),
                        'instruction': ep['instruction'],
                        'success': ep.get('success', False),
                        'steps': ep.get('steps', []),
                        'raw_data': ep
                    }
                    for score, ep in similarities[:limit]
                ]
            except Exception as e:
                logger.error(f"Embedding-based retrieval failed: {e}")
        
        # Fallback: simple text matching
        return self._simple_text_search(instruction, limit)
    
    def _simple_text_search(self, instruction: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback text-based similarity search"""
        instruction_lower = instruction.lower()
        
        matches = []
        for episode in self.episodes:
            ep_instruction = episode.get('instruction', '').lower()
            # Simple word overlap score
            query_words = set(instruction_lower.split())
            ep_words = set(ep_instruction.split())
            overlap = len(query_words & ep_words)
            
            if overlap > 0:
                matches.append((overlap / len(query_words | ep_words), episode))
        
        matches.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                'score': float(score),
                'instruction': ep['instruction'],
                'success': ep.get('success', False),
                'steps': ep.get('steps', []),
                'raw_data': ep
            }
            for score, ep in matches[:limit]
        ]
    
    def _generate_episode_id(self, episode_data: Dict[str, Any]) -> str:
        """Generate unique episode ID"""
        text = f"{episode_data['instruction']}_{episode_data.get('timestamp', '')}"
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def get_success_rate(self, instruction_pattern: Optional[str] = None) -> float:
        """Calculate success rate for episodes matching pattern"""
        if not self.episodes:
            return 0.0
        
        relevant_episodes = self.episodes
        if instruction_pattern:
            pattern_lower = instruction_pattern.lower()
            relevant_episodes = [
                ep for ep in self.episodes 
                if pattern_lower in ep.get('instruction', '').lower()
            ]
        
        if not relevant_episodes:
            return 0.0
        
        successes = sum(1 for ep in relevant_episodes if ep.get('success', False))
        return successes / len(relevant_episodes)
