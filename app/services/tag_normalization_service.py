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
            items: List of tags with 'label' and 'prob' keys (optionally 'tag' for raw PANNs label)
            opts: Normalization options
        
        Returns:
            List of normalized tags preserving 'tag' (raw), with 'label' and 'prob'
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
        
        # 1) Sort by probability + floor (no topK cut)
        sorted_items = sorted(items, key=lambda x: x.get('prob', 0), reverse=True)
        ps = [max(item.get('prob', 0), self.EPS) for item in sorted_items]
        
        if not ps:
            return []
        
        # 2) Log-scale → softmax with temperature (mix inside group)
        xs = [math.log(p + self.EPS) for p in ps]
        q = self._softmax(xs, temperature)
        
        # 3) Build result: keep only 'tag' (raw) and 'prob' for internal processing
        result = []
        for i, item in enumerate(sorted_items):
            result.append({
                'tag': item.get('tag', ''),
                'prob': q[i]
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
    
    def prepare_for_client(self, tags_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepare normalized tags for client output: convert 'tag' to 'label' and remove 'tag'
        
        Args:
            tags_dict: Internal tags with 'tag' and 'prob' fields
            
        Returns:
            Client-ready tags with 'label' and 'prob' fields
        """
        from ..services.panns_service import PANNsService
        
        # Create PANNsService instance for prettification
        panns_service = PANNsService()
        
        result = {}
        for group_name, items in tags_dict.items():
            client_items = []
            for item in items:
                raw_tag = item.get('tag', '')
                prob = item.get('prob', 0.0)
                
                # Convert raw tag to pretty label
                pretty_label = panns_service.prettify_label(raw_tag, group_name)
                
                client_items.append({
                    'label': pretty_label,
                    'prob': prob
                })
            
            result[group_name] = client_items
        
        return result
