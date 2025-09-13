#!/usr/bin/env python3
"""
Skills Embedder using EmbeddingGemma
Embeds skills sections from bio JSON using EmbeddingGemma clustering mode.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SkillsEmbedder:
    """Embeds and clusters skills using EmbeddingGemma."""
    
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.skills_data = []
        
    def setup_model(self):
        """Initialize EmbeddingGemma model for clustering."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[MODEL] Loading EmbeddingGemma model...")
            self.model = SentenceTransformer("google/embeddinggemma-300m")
            logger.info("[MODEL] Model loaded successfully")
            
            # Check available prompts
            logger.info(f"[MODEL] Available prompts: {list(self.model.prompts.keys())}")
            
        except ImportError as e:
            logger.error("[ERROR] Missing dependencies. Install with:")
            logger.error("pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview")
            logger.error("pip install sentence-transformers>=5.0.0")
            raise e
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            raise e
    
    def extract_skills_text(self, bio: Dict[str, Any]) -> str:
        """Extract skills section as formatted text for embedding."""
        skills = bio.get('skills', [])
        if not skills:
            return ""
        
        # Format skills as structured text for clustering
        skills_parts = []
        for skill in skills:
            skill_name = skill.get('skill', '')
            years = skill.get('years', 0)
            proficiency = skill.get('proficiency', 'Unknown')
            skills_parts.append(f"{skill_name} ({years} years, {proficiency})")
        
        # Include role and industry context for better embeddings
        role = bio.get('role', '')
        industry = bio.get('industry', '')
        specializations = bio.get('specializations', [])
        
        context_text = f"Role: {role}, Industry: {industry}"
        if specializations:
            context_text += f", Specializations: {', '.join(specializations)}"
        
        skills_text = f"{context_text}. Skills: {', '.join(skills_parts)}"
        return skills_text
    
    def load_and_extract_skills(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load bios and extract skills data."""
        logger.info(f"[LOAD] Loading bios from: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                bios = json.load(f)
            logger.info(f"[LOAD] Loaded {len(bios)} bios")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load JSON: {e}")
            raise e
        
        skills_data = []
        for idx, bio in enumerate(bios):
            skills_text = self.extract_skills_text(bio)
            if skills_text:
                skills_data.append({
                    'bio_index': idx,
                    'role': bio.get('role', ''),
                    'industry': bio.get('industry', ''),
                    'skills_text': skills_text,
                    'skills_count': len(bio.get('skills', [])),
                    'total_experience': bio.get('total_experience_years', 0)
                })
        
        logger.info(f"[EXTRACT] Extracted {len(skills_data)} skill profiles")
        self.skills_data = skills_data
        return skills_data
    
    def embed_skills(self, truncate_dim: Optional[int] = None):
        """Generate embeddings for skills using clustering prompt."""
        if not self.model:
            self.setup_model()
        
        if not self.skills_data:
            logger.error("[ERROR] No skills data loaded")
            return
        
        logger.info(f"[EMBED] Generating embeddings for {len(self.skills_data)} skill profiles")
        
        # Extract skills texts
        texts = [item['skills_text'] for item in self.skills_data]
        
        # Generate embeddings using clustering prompt
        logger.info("[EMBED] Using 'Clustering' prompt for EmbeddingGemma")
        
        embed_kwargs = {'prompt_name': 'Clustering'}
        if truncate_dim:
            embed_kwargs['truncate_dim'] = truncate_dim
            logger.info(f"[EMBED] Truncating embeddings to {truncate_dim} dimensions")
        
        embeddings = self.model.encode(texts, **embed_kwargs)
        self.embeddings = embeddings
        
        logger.info(f"[EMBED] Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def cluster_skills(self, n_clusters_range: range = range(3, 15)) -> Dict[str, Any]:
        """Perform K-means clustering on skills embeddings."""
        if self.embeddings is None:
            logger.error("[ERROR] No embeddings available. Run embed_skills first.")
            return {}
        
        logger.info(f"[CLUSTER] Finding optimal number of clusters in range {n_clusters_range}")
        
        # Find optimal number of clusters using silhouette score
        silhouette_scores = []
        inertias = []
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                silhouette_avg = silhouette_score(self.embeddings, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                inertias.append(kmeans.inertia_)
            else:
                silhouette_scores.append(0)
                inertias.append(float('inf'))
        
        # Find optimal number of clusters
        optimal_idx = np.argmax(silhouette_scores)
        optimal_clusters = list(n_clusters_range)[optimal_idx]
        
        logger.info(f"[CLUSTER] Optimal clusters: {optimal_clusters} (silhouette score: {silhouette_scores[optimal_idx]:.3f})")
        
        # Perform final clustering
        final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(self.embeddings)
        
        # Add cluster labels to skills data
        for i, item in enumerate(self.skills_data):
            item['cluster'] = int(cluster_labels[i])
        
        return {
            'optimal_clusters': optimal_clusters,
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'cluster_labels': cluster_labels.tolist(),
            'kmeans_model': final_kmeans
        }
    
    def analyze_clusters(self, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and summarize the skill clusters."""
        if not clustering_results:
            return {}
        
        logger.info("[ANALYZE] Analyzing skill clusters")
        
        # Group by clusters
        clusters_analysis = {}
        for cluster_id in range(clustering_results['optimal_clusters']):
            cluster_items = [item for item in self.skills_data if item['cluster'] == cluster_id]
            
            # Analyze cluster characteristics
            roles = [item['role'] for item in cluster_items]
            industries = [item['industry'] for item in cluster_items]
            
            # Count frequencies
            role_counts = pd.Series(roles).value_counts().to_dict()
            industry_counts = pd.Series(industries).value_counts().to_dict()
            
            avg_experience = np.mean([item['total_experience'] for item in cluster_items])
            avg_skills_count = np.mean([item['skills_count'] for item in cluster_items])
            
            clusters_analysis[cluster_id] = {
                'size': len(cluster_items),
                'top_roles': dict(list(role_counts.items())[:5]),
                'top_industries': dict(list(industry_counts.items())[:5]),
                'avg_experience_years': round(avg_experience, 1),
                'avg_skills_count': round(avg_skills_count, 1),
                'sample_profiles': [item['skills_text'][:200] + "..." for item in cluster_items[:3]]
            }
        
        return clusters_analysis
    
    def save_results(self, output_path: Path, clustering_results: Dict[str, Any], 
                    clusters_analysis: Dict[str, Any]):
        """Save embeddings and clustering results."""
        logger.info(f"[SAVE] Saving results to: {output_path}")
        
        results = {
            'metadata': {
                'total_profiles': len(self.skills_data),
                'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
                'optimal_clusters': clustering_results.get('optimal_clusters', 0)
            },
            'skills_data': self.skills_data,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else [],
            'clustering_results': {k: v for k, v in clustering_results.items() if k != 'kmeans_model'},
            'clusters_analysis': clusters_analysis
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[SAVE] Results saved successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save results: {e}")


def main():
    """Main execution function."""
    
    logger.info("[START] Starting Skills Embedding with EmbeddingGemma")
    
    project_dir = Path(__file__).parent
    input_path = project_dir / "enhanced_bios.json"
    output_path = project_dir / "skills_embeddings.json"
    
    if not input_path.exists():
        logger.error(f"[ERROR] Input file not found: {input_path}")
        return
    
    # Initialize embedder
    embedder = SkillsEmbedder()
    
    try:
        # Load and extract skills
        embedder.load_and_extract_skills(input_path)
        
        # Generate embeddings (with dimension reduction for efficiency)
        embedder.embed_skills(truncate_dim=256)
        
        # Perform clustering
        clustering_results = embedder.cluster_skills()
        
        # Analyze clusters
        clusters_analysis = embedder.analyze_clusters(clustering_results)
        
        # Print summary
        logger.info(f"[SUMMARY] Found {clustering_results['optimal_clusters']} skill clusters")
        for cluster_id, analysis in clusters_analysis.items():
            logger.info(f"[CLUSTER {cluster_id}] Size: {analysis['size']}, "
                       f"Avg Experience: {analysis['avg_experience_years']}y, "
                       f"Top Role: {list(analysis['top_roles'].keys())[0] if analysis['top_roles'] else 'N/A'}")
        
        # Save results
        embedder.save_results(output_path, clustering_results, clusters_analysis)
        
        logger.info(f"[COMPLETE] Skills embedding and clustering complete!")
        logger.info(f"[OUTPUT] Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"[ERROR] Processing failed: {e}")
        raise e


if __name__ == "__main__":
    main()
