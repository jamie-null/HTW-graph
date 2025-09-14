#!/usr/bin/env python3
"""
Local Skills Embedder using llama.cpp server
Embeds skills sections from bio JSON using local llama.cpp embedding server.
"""

import json
import numpy as np
import pandas as pd
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import signal
import sys

# Configure logging with DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalSkillsEmbedder:
    """Embeds and clusters skills using local llama.cpp server."""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.embedding_endpoint = f"{server_url}/embedding"
        self.embeddings = None
        self.skills_data = []
        self.csv_data = None
        self.shutdown_requested = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("[SHUTDOWN] Shutdown signal received. Saving progress...")
        self.shutdown_requested = True
        
    def test_server_connection(self) -> bool:
        """Test connection to local llama.cpp server."""
        try:
            logger.info(f"[SERVER] Testing connection to {self.server_url}")
            test_response = requests.post(
                self.embedding_endpoint,
                json={"content": "test connection"},
                timeout=10
            )
            if test_response.status_code == 200:
                logger.info("[SERVER] Connection successful")
                return True
            else:
                logger.error(f"[SERVER] Server returned status code: {test_response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"[SERVER] Connection failed: {e}")
            return False
    
    def get_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Get embedding for text from local server."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.embedding_endpoint,
                    json={"content": text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"[DEBUG] Raw response type: {type(result)}")
                    logger.debug(f"[DEBUG] Raw response keys: {result.keys() if isinstance(result, dict) else 'not dict'}")
                    
                    # Handle llama.cpp specific format: [{'index': 0, 'embedding': [...]}]
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and 'embedding' in result[0]:
                            embedding_data = result[0]['embedding']
                            
                            # Handle nested embedding format: [[actual_embedding_values]]
                            if isinstance(embedding_data, list) and len(embedding_data) > 0 and isinstance(embedding_data[0], list):
                                logger.debug("Found nested embedding format, extracting inner list")
                                embedding_data = embedding_data[0]
                            
                            if isinstance(embedding_data, list) and all(isinstance(x, (int, float)) for x in embedding_data):
                                logger.debug(f"[SUCCESS] Valid embedding extracted with {len(embedding_data)} dimensions")
                                return embedding_data
                        elif all(isinstance(x, (int, float)) for x in result):
                            return result
                    elif isinstance(result, dict):
                        # Try common embedding field names
                        for key in ['embedding', 'data', 'results', 'vector', 'embeddings']:
                            if key in result:
                                embedding_data = result[key]
                                if isinstance(embedding_data, list) and all(isinstance(x, (int, float)) for x in embedding_data):
                                    return embedding_data
                        
                        # If it's a nested structure, try to extract
                        logger.debug(f"[DEBUG] Full response structure: {result}")
                        
                        # Handle llama.cpp format: list with index and embedding keys
                        if isinstance(result, list) and len(result) > 0:
                            first_item = result[0]
                            if isinstance(first_item, dict) and 'embedding' in first_item:
                                logger.debug(f"Found llama.cpp format with keys: {first_item.keys()}")
                                embedding = first_item['embedding']
                                logger.debug(f"Embedding type: {type(embedding)}")
                                logger.debug(f"Embedding length: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
                                
                                # Handle nested embedding format: [[actual_embedding_values]]
                                if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                                    logger.debug("Found nested embedding format, extracting inner list")
                                    embedding = embedding[0]
                                
                                # Validate that embedding is a list of numbers
                                if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                                    logger.debug(f"[SUCCESS] Valid embedding extracted with {len(embedding)} dimensions")
                                    return embedding
                                else:
                                    logger.error("Embedding is not a valid numeric list")
                                    return None
                        # Sometimes responses are wrapped like {"data": [{"embedding": [...]}]}
                        if 'data' in result and isinstance(result['data'], list) and len(result['data']) > 0:
                            first_item = result['data'][0]
                            if isinstance(first_item, dict):
                                for key in ['embedding', 'vector', 'embeddings']:
                                    if key in first_item:
                                        embedding_data = first_item[key]
                                        if isinstance(embedding_data, list) and all(isinstance(x, (int, float)) for x in embedding_data):
                                            return embedding_data
                    
                    logger.error(f"[EMBED] Cannot extract embedding from response format")
                    logger.error(f"[EMBED] Response: {str(result)[:500]}...")
                    return None
                else:
                    logger.warning(f"[EMBED] Server error {response.status_code}, attempt {attempt + 1}")
                    logger.debug(f"[DEBUG] Response text: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"[EMBED] Request failed on attempt {attempt + 1}: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(1)  # Brief delay before retry
        
        logger.error(f"[EMBED] Failed to get embedding after {max_retries} attempts")
        return None
    
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
        
        context_text = f"Professional with role: {role} in {industry} industry"
        if specializations:
            context_text += f", specializing in {', '.join(specializations)}"
        
        skills_text = f"{context_text}. Core skills and experience: {', '.join(skills_parts)}"
        return skills_text
    
    def load_csv_data(self, csv_path: Path) -> bool:
        """Load CSV data for merging with bio data."""
        try:
            logger.info(f"[CSV] Loading CSV from: {csv_path}")
            self.csv_data = pd.read_csv(csv_path)
            logger.info(f"[CSV] Loaded CSV with {len(self.csv_data)} rows")
            logger.info(f"[CSV] CSV columns: {list(self.csv_data.columns)}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to load CSV: {e}")
            return False
    
    def merge_csv_with_bio(self, bio: Dict[str, Any], bio_index: int) -> Dict[str, Any]:
        """Merge CSV data with a single bio entry."""
        if self.csv_data is None or bio_index >= len(self.csv_data):
            return bio
            
        try:
            # Get the corresponding CSV row by index (assuming sequential matching)
            csv_row = self.csv_data.iloc[bio_index]
            
            # Add CSV columns to bio, preserving existing bio data
            enhanced_bio = bio.copy()
            
            # Add timestamp and location data
            if 'CONFIRM_TIME' in csv_row:
                enhanced_bio['confirm_time'] = str(csv_row['CONFIRM_TIME'])
            if 'LAST_CHANGED' in csv_row:
                enhanced_bio['last_changed'] = str(csv_row['LAST_CHANGED'])
            if 'Location' in csv_row:
                enhanced_bio['csv_location'] = str(csv_row['Location'])
            if 'Role / Job Title' in csv_row:
                enhanced_bio['csv_role'] = str(csv_row['Role / Job Title'])
            if 'Industry' in csv_row:
                enhanced_bio['csv_industry'] = str(csv_row['Industry'])
                
            return enhanced_bio
        except Exception as e:
            logger.warning(f"[MERGE] Failed to merge CSV data for bio {bio_index}: {e}")
            return bio
    
    def load_existing_embeddings(self, output_path: Path) -> Dict[str, Any]:
        """Load existing embeddings for resume capability."""
        if not output_path.exists():
            logger.info("[RESUME] No existing embeddings found. Starting fresh.")
            return {}
            
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            processed_count = len(existing_data.get('embeddings', []))
            logger.info(f"[RESUME] Found {processed_count} existing embeddings")
            return existing_data
        except Exception as e:
            logger.warning(f"[RESUME] Failed to load existing embeddings: {e}")
            return {}
    
    def load_and_extract_skills(self, json_path: Path, csv_path: Path, output_path: Path) -> List[Dict[str, Any]]:
        """Load bios, merge with CSV, and extract skills text for embedding."""
        logger.info(f"[LOAD] Loading bios from: {json_path}")
        
        # Load CSV data
        if not self.load_csv_data(csv_path):
            return []
        
        # Load existing embeddings for resume capability
        existing_data = self.load_existing_embeddings(output_path)
        existing_embeddings = existing_data.get('embeddings', [])
        existing_skills = existing_data.get('skills_data', [])
        
        with open(json_path, 'r', encoding='utf-8') as f:
            bios = json.load(f)
        
        logger.info(f"[LOAD] Loaded {len(bios)} bios")
        
        # Resume from where we left off
        start_index = len(existing_embeddings)
        if start_index > 0:
            logger.info(f"[RESUME] Resuming from bio index {start_index}")
            self.skills_data = existing_skills.copy()
        else:
            self.skills_data = []
        
        # First, update existing entries to add missing CSV data
        if existing_skills:
            logger.info(f"[UPDATE] Adding CSV data to {len(existing_skills)} existing entries")
            for existing_entry in self.skills_data:
                bio_idx = existing_entry['bio_index']
                if bio_idx < len(bios):
                    logger.info(f"[UPDATE] Row {bio_idx + 1}: Updating existing entry with CSV data")
                    enhanced_bio = self.merge_csv_with_bio(bios[bio_idx], bio_idx)
                    # Add missing CSV fields to existing entry
                    existing_entry['confirm_time'] = enhanced_bio.get('confirm_time', '')
                    existing_entry['last_changed'] = enhanced_bio.get('last_changed', '')
                    existing_entry['csv_location'] = enhanced_bio.get('csv_location', '')
                    existing_entry['csv_role'] = enhanced_bio.get('csv_role', '')
                    existing_entry['csv_industry'] = enhanced_bio.get('csv_industry', '')
        
        # Process remaining bios
        for i in range(start_index, len(bios)):
            if self.shutdown_requested:
                logger.info(f"[SHUTDOWN] Stopping at bio {i}")
                break
                
            bio = bios[i]
            logger.info(f"[PROCESS] Row {i + 1}/{len(bios)}: Processing bio for {bio.get('role', 'Unknown Role')}")
            
            # Merge with CSV data
            enhanced_bio = self.merge_csv_with_bio(bio, i)
            
            skills_text = self.extract_skills_text(enhanced_bio)
            if skills_text:  # Only process bios with skills
                self.skills_data.append({
                    'bio_index': i,
                    'role': enhanced_bio.get('role', ''),
                    'industry': enhanced_bio.get('industry', ''),
                    'skills_text': skills_text,
                    'skills_count': len(enhanced_bio.get('skills', [])),
                    'total_experience': sum(skill.get('years', 0) for skill in enhanced_bio.get('skills', [])),
                    'confirm_time': enhanced_bio.get('confirm_time', ''),
                    'last_changed': enhanced_bio.get('last_changed', ''),
                    'csv_location': enhanced_bio.get('csv_location', ''),
                    'csv_role': enhanced_bio.get('csv_role', ''),
                    'csv_industry': enhanced_bio.get('csv_industry', '')
                })
                logger.debug(f"[EXTRACT] Row {i + 1}: Added skills profile with CSV data")
            else:
                logger.warning(f"[SKIP] Row {i + 1}: No skills found in bio")
        
        logger.info(f"[EXTRACT] Extracted {len(self.skills_data)} skill profiles (including {len(existing_skills)} existing)")
        
        # Store existing embeddings
        if existing_embeddings:
            self.embeddings = np.array(existing_embeddings)
            logger.info(f"[RESUME] Loaded {len(existing_embeddings)} existing embeddings")
        
        return self.skills_data
    
    def embed_skills(self, output_path: Path, batch_size: int = 10, save_progress_every: int = 5) -> Optional[np.ndarray]:
        """Generate embeddings for skills using local server with resume capability."""
        if not self.skills_data:
            logger.error("[ERROR] No skills data loaded")
            return None
        
        # Determine how many embeddings already exist
        existing_count = len(self.embeddings) if self.embeddings is not None else 0
        total_needed = len(self.skills_data)
        remaining = total_needed - existing_count
        
        if remaining <= 0:
            logger.info(f"[EMBED] All {total_needed} embeddings already exist")
            return self.embeddings
        
        logger.info(f"[EMBED] Generating embeddings for {remaining} remaining skill profiles ({existing_count} already exist)")
        logger.info(f"[EMBED] Processing in batches of {batch_size}")
        
        # Start with existing embeddings
        embeddings_list = self.embeddings.tolist() if self.embeddings is not None else []
        embedding_dim = len(embeddings_list[0]) if embeddings_list else None
        failed_count = 0
        
        # Process only the remaining profiles
        profiles_to_process = self.skills_data[existing_count:]
        
        for i in range(0, len(profiles_to_process), batch_size):
            if self.shutdown_requested:
                logger.info(f"[SHUTDOWN] Stopping embedding at batch {i//batch_size + 1}")
                break
                
            batch = profiles_to_process[i:i + batch_size]
            batch_end = min(i + batch_size, len(profiles_to_process))
            
            logger.info(f"[EMBED] Processing batch {i//batch_size + 1}: items {existing_count + i + 1}-{existing_count + batch_end}")
            
            for j, profile in enumerate(batch):
                bio_idx = profile['bio_index']
                logger.info(f"[EMBED] Row {bio_idx + 1}: Embedding NEW profile for {profile['role'][:50]}...")
                
                embedding = self.get_embedding(profile['skills_text'])
                
                if embedding is not None:
                    if embedding_dim is None:
                        embedding_dim = len(embedding)
                        logger.info(f"[EMBED] Determined embedding dimension: {embedding_dim}")
                    elif len(embedding) != embedding_dim:
                        logger.warning(f"[EMBED] Row {bio_idx + 1}: Dimension mismatch - expected {embedding_dim}, got {len(embedding)}")
                        embedding = None
                
                if embedding is not None:
                    embeddings_list.append(embedding)
                    logger.debug(f"[SUCCESS] Row {bio_idx + 1}: Embedding generated successfully")
                else:
                    # Use zero vector as placeholder for failed embeddings
                    if embedding_dim is not None:
                        embeddings_list.append([0.0] * embedding_dim)
                    else:
                        logger.error("[ERROR] Cannot determine embedding dimension - no valid embeddings yet")
                        return None
                    failed_count += 1
                    logger.warning(f"[SKIP] Row {bio_idx + 1}: Failed to embed profile")
                
                # Small delay to be respectful to server
                time.sleep(0.1)
            
            batch_num = i // batch_size + 1
            if batch_num % save_progress_every == 0:
                temp_embeddings = np.array(embeddings_list)
                self.embeddings = temp_embeddings
                self.save_results(output_path)
                logger.info(f"[PROGRESS] Saved progress: {len(embeddings_list)}/{total_needed} embeddings to file")
        
        if failed_count > 0:
            logger.warning(f"[EMBED] {failed_count} embeddings failed and were replaced with zeros")
        
        if not embeddings_list:
            logger.error("[ERROR] No valid embeddings generated")
            return None
        
        try:
            logger.info(f"[CONVERT] Converting {len(embeddings)} embeddings to numpy array")
            logger.debug(f"[DEBUG] First embedding type: {type(embeddings[0])}, length: {len(embeddings[0]) if embeddings[0] else 'None'}")
            self.embeddings = np.array(embeddings)
            logger.info(f"[EMBED] Generated embeddings shape: {self.embeddings.shape}")
            return self.embeddings
        except Exception as e:
            logger.error(f"[ERROR] Failed to convert embeddings to numpy array: {e}")
            logger.error(f"[DEBUG] Embeddings sample: {embeddings[0] if embeddings else 'None'}")
            return None
    
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
    
    def save_results(self, output_path: Path):
        """Save embeddings and skills data with CSV time fields."""
        logger.info(f"[SAVE] Saving {len(self.skills_data)} profiles with embeddings to: {output_path}")
        
        # Verify CSV data is included
        csv_fields_count = sum(1 for entry in self.skills_data if entry.get('confirm_time') or entry.get('last_changed'))
        logger.info(f"[SAVE] {csv_fields_count}/{len(self.skills_data)} entries have CSV time data")
        
        results = {
            'metadata': {
                'total_profiles': len(self.skills_data),
                'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
                'server_url': self.server_url,
                'csv_fields_included': csv_fields_count > 0
            },
            'skills_data': self.skills_data,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else []
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[SAVE] Results saved successfully with {csv_fields_count} CSV-enhanced entries")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save results: {e}")


def main():
    """Main execution function."""
    
    logger.info("[START] Starting Local Skills Embedding with llama.cpp server")
    
    project_dir = Path(__file__).parent
    bios_path = project_dir / "generated_bios.json"
    csv_path = project_dir / "HTW Network Data - Pseunom Subset of ~2.5K People - HTW2025Audience.csv"
    output_path = project_dir / "local_skills_embeddings.json"
    
    if not bios_path.exists():
        logger.error(f"[ERROR] Bios file not found: {bios_path}")
        return
        
    if not csv_path.exists():
        logger.error(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    # Initialize embedder
    embedder = LocalSkillsEmbedder()
    
    # Test server connection
    if not embedder.test_server_connection():
        logger.error("[ERROR] Cannot connect to llama.cpp server at localhost:8080")
        logger.error("Please ensure your llama.cpp server is running with embedding support")
        return
    
    try:
        # Load and extract skills with CSV merging and resume capability  
        skills_data = embedder.load_and_extract_skills(bios_path, csv_path, output_path)
        if not skills_data:
            logger.error("[ERROR] Failed to load and extract skills")
            return
        
        # Generate embeddings using local server with resume capability
        embeddings = embedder.embed_skills(output_path, save_progress_every=3)
        
        if embeddings is not None:
            logger.info(f"[SUMMARY] Successfully generated {len(embeddings)} embeddings")
            logger.info(f"[SUMMARY] Embedding dimensions: {embeddings.shape[1]}")
            
            # Save final results
            embedder.save_results(output_path)
            
            logger.info(f"[COMPLETE] Local skills embedding complete!")
            logger.info(f"[OUTPUT] Results saved to: {output_path}")
        else:
            logger.error("[ERROR] Failed to generate embeddings")
        
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] Interrupted by user")
        # Save progress before exiting
        if embedder.embeddings is not None:
            embedder.save_results(output_path)
            logger.info(f"[SHUTDOWN] Progress saved to: {output_path}")
    except Exception as e:
        logger.error(f"[ERROR] Processing failed: {e}")
        # Save progress on error too
        if embedder.embeddings is not None:
            embedder.save_results(output_path)
            logger.info(f"[ERROR] Progress saved to: {output_path}")
        raise e


if __name__ == "__main__":
    main()
