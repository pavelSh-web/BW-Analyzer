"""Service for tag probability normalization"""

import math
from typing import List, Dict, Any, Optional


class TagNormalizationService:
    """Service for normalizing tag probabilities using softmax and group strength"""
    
    EPS = 1e-12
    
    def __init__(self):
        pass
    
    def _softmax(self, xs: List[float], temperature: float = 1.0) -> List[float]:
        """Compute softmax with temperature"""
        scaled = [x / temperature for x in xs]
        max_val = max(scaled)
        exps = [math.exp(x - max_val) for x in scaled]
        sum_exps = sum(exps) + self.EPS
        return [v / sum_exps for v in exps]
    
    def _entropy(self, p: List[float]) -> float:
        """Compute entropy of probability distribution"""
        return -sum(v * math.log(v + self.EPS) if v > 0 else 0 for v in p)
    
    def normalize_group(self, items: List[Dict[str, Any]], opts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Normalize probabilities within a group
        
        Args:
            items: List of tags with 'label' and 'prob' keys
            opts: Normalization options
        
        Returns:
            List of normalized tags with 'label', 'raw', 'mix', 'score' keys
        """
        if not items:
            return []
        
        # Default options (topK removed — use all items)
        default_opts = {
            'temperature': 1.2
        }
        
        if opts:
            default_opts.update(opts)
        
        temperature = default_opts['temperature']
        
        # 1) Сортировка по вероятности + floor (без отсечения по topK)
        sorted_items = sorted(items, key=lambda x: x.get('prob', 0), reverse=True)
        ps = [max(item.get('prob', 0), self.EPS) for item in sorted_items]
        
        if not ps:
            return []
        
        # 2) Лог-шкала → softmax с температурой (микс внутри группы)
        xs = [math.log(p + self.EPS) for p in ps]
        q = self._softmax(xs, temperature)
        
        # Формируем результат: используем mix как prob
        result = []
        for i, item in enumerate(sorted_items):
            result.append({
                'label': item.get('label', ''),
                'prob': q[i]      # распределение внутри группы (сумма=1)
            })
        
        return result
    
    def normalize_tags_dict(self, tags_dict: Dict[str, List[Dict[str, Any]]], 
                           opts: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Normalize tags for all groups
        
        Args:
            tags_dict: Dictionary with group names as keys and lists of tags as values
            opts: Normalization options
        
        Returns:
            Dictionary with normalized tags for each group
        """
        result = {}
        
        for group_name, items in tags_dict.items():
            if items:
                result[group_name] = self.normalize_group(items, opts)
            else:
                result[group_name] = []
        
        return result
