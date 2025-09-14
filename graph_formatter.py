#!/usr/bin/env python3
"""
Graph Formatter for HTW Network
Converts local_skills_embeddings.json to network graph format with Watts-Strogatz model
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphFormatter:
    """Formats skills embedding data into network graph format."""
    
    def __init__(self):
        self.skills_data = []
        self.embeddings = []
        self.skill_lexicon = []
        self.nodes = []
        self.links = []
        
    def load_embeddings_data(self, input_path: Path) -> bool:
        """Load the local skills embeddings JSON data."""
        try:
            logger.info(f"[LOAD] Loading embeddings data from: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.skills_data = data.get('skills_data', [])
            self.embeddings = data.get('embeddings', [])
            
            logger.info(f"[LOAD] Loaded {len(self.skills_data)} skill profiles")
            logger.info(f"[LOAD] Loaded {len(self.embeddings)} embeddings ({len(self.embeddings[0])} dimensions)")
            
            if len(self.skills_data) != len(self.embeddings):
                logger.warning(f"[LOAD] Mismatch: {len(self.skills_data)} profiles vs {len(self.embeddings)} embeddings")
            
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to load embeddings data: {e}")
            return False
    
    def extract_skill_lexicon(self) -> List[str]:
        """Extract unique skills from all profiles to create lexicon."""
        logger.info("[LEXICON] Extracting skill lexicon from profiles")
        
        all_skills = []
        
        for profile in self.skills_data:
            skills_text = profile.get('skills_text', '')
            
            # Extract skills using regex pattern for "Skill Name (X years, Level)" format
            skill_pattern = r'([^(]+?)\s*\(\d+\s*years?,\s*[^)]+\)'
            matches = re.findall(skill_pattern, skills_text)
            
            for match in matches:
                skill = match.strip()
                # Clean up skill name
                skill = re.sub(r'^[:\-,.\s]+|[:\-,.\s]+$', '', skill)
                if skill and len(skill) > 1:
                    all_skills.append(skill)
        
        # Count frequency and take most common skills
        skill_counts = Counter(all_skills)
        
        # Get top skills (minimum 2 occurrences)
        common_skills = [skill for skill, count in skill_counts.most_common() if count >= 2]
        
        # Add some single-occurrence skills for diversity (up to 50 total)
        if len(common_skills) < 50:
            rare_skills = [skill for skill, count in skill_counts.most_common() if count == 1]
            needed = min(50 - len(common_skills), len(rare_skills))
            common_skills.extend(rare_skills[:needed])
        
        self.skill_lexicon = common_skills[:50]  # Limit to 50 skills for manageable vectors
        
        logger.info(f"[LEXICON] Created skill lexicon with {len(self.skill_lexicon)} skills")
        logger.info(f"[LEXICON] Top 10 skills: {self.skill_lexicon[:10]}")
        
        return self.skill_lexicon
    
    def extract_person_skills(self, skills_text: str) -> List[str]:
        """Extract individual skills from a person's skills_text."""
        person_skills = []
        
        # Extract skills using regex pattern
        skill_pattern = r'([^(]+?)\s*\(\d+\s*years?,\s*[^)]+\)'
        matches = re.findall(skill_pattern, skills_text)
        
        for match in matches:
            skill = match.strip()
            skill = re.sub(r'^[:\-,.\s]+|[:\-,.\s]+$', '', skill)
            if skill and skill in self.skill_lexicon:
                person_skills.append(skill)
        
        return person_skills
    
    def create_skill_vector(self, person_skills: List[str]) -> List[float]:
        """Create normalized skill vector based on lexicon."""
        vector = [0.0] * len(self.skill_lexicon)
        
        if not person_skills:
            return vector
        
        # Set 1.0 for skills the person has, then normalize
        for skill in person_skills:
            if skill in self.skill_lexicon:
                idx = self.skill_lexicon.index(skill)
                vector[idx] = 1.0
        
        # Normalize to unit vector if any skills present
        magnitude = np.sqrt(sum(x**2 for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector
    
    def generate_person_name(self, role: str, bio_index: int) -> str:
        """Generate a person name from role and index."""
        # Extract key words from role
        role_words = re.findall(r'\b[A-Z][a-z]+', role.replace('&', 'and'))
        
        # Common first names
        first_names = ["Alex", "Morgan", "Taylor", "Jordan", "Casey", "Riley", "Avery", "Quinn", 
                      "Sage", "River", "Rowan", "Phoenix", "Skyler", "Cameron", "Emery", "Finley"]
        
        # Generate consistent name based on index
        first_name = first_names[bio_index % len(first_names)]
        
        # Generate last name from role or use generic
        if role_words:
            last_name = role_words[0][:8]  # Truncate if too long
        else:
            last_names = ["Chen", "Martinez", "Kim", "Thompson", "Johnson", "Williams", "Brown", "Davis"]
            last_name = last_names[bio_index % len(last_names)]
        
        return f"{first_name} {last_name}"
    
    def generate_org_name(self, industry: str) -> str:
        """Generate organization name from industry."""
        # Industry keyword mapping
        org_mapping = {
            "Technology": "TechCorp",
            "Healthcare": "MedGroup", 
            "Education": "EduLab",
            "Finance": "FinTech",
            "Consulting": "ConsultPro",
            "Government": "GovSec",
            "Media": "MediaHub",
            "Gaming": "GameDev",
            "Construction": "BuildCorp",
            "Energy": "EnergyTech"
        }
        
        # Find matching keyword
        for keyword, org in org_mapping.items():
            if keyword.lower() in industry.lower():
                return org
        
        # Default organization
        return "InnovaCorp"
    
    def create_nodes(self) -> List[Dict[str, Any]]:
        """Convert skills data to node format using real embeddings."""
        logger.info("[NODES] Creating nodes from skills data with real embeddings")
        
        nodes = []
        
        for i, profile in enumerate(self.skills_data):
            # Extract person skills for display
            person_skills = self.extract_person_skills(profile.get('skills_text', ''))
            
            # Use actual embedding as skill vector
            if i < len(self.embeddings):
                skill_vector = self.embeddings[i]
                # Round to 3 decimal places and ensure it's a list
                skill_vector = [round(float(x), 3) for x in skill_vector]
            else:
                # Fallback to zero vector if no embedding
                skill_vector = [0.0] * 768
                logger.warning(f"[NODES] No embedding found for profile {i}, using zero vector")
            
            # Generate names and org
            person_name = self.generate_person_name(profile.get('role', ''), profile.get('bio_index', i))
            org_name = self.generate_org_name(profile.get('industry', ''))
            
            # Simplify role for display
            role = profile.get('role', 'Professional')
            role_simplified = role.split(',')[0].split('&')[0].strip()
            if len(role_simplified) > 20:
                role_simplified = role_simplified[:17] + "..."
            
            node = {
                "id": f"person_{profile.get('bio_index', i)}",
                "name": person_name,
                "role": role_simplified,
                "org": org_name,
                "skills": person_skills[:5],  # Limit to top 5 skills for readability
                "skill_vector": skill_vector,  # Use real 768-dimensional embedding
                "isFocus": i == 0  # First person is focus
            }
            
            nodes.append(node)
        
        logger.info(f"[NODES] Created {len(nodes)} nodes with {len(skill_vector)}-dimensional embeddings")
        self.nodes = nodes
        return nodes
    
    def sort_by_confirm_time(self) -> List[Dict[str, Any]]:
        """Sort nodes by confirm_time for lattice ordering."""
        logger.info("[SORT] Sorting nodes by confirm_time")
        
        # Parse confirm_time and sort
        def parse_time(profile_data):
            confirm_time = profile_data.get('confirm_time', '')
            if confirm_time:
                try:
                    return datetime.strptime(confirm_time, '%Y-%m-%d %H:%M:%S')
                except:
                    pass
            return datetime.min
        
        # Sort skills_data by time, then update nodes accordingly
        sorted_profiles = sorted(self.skills_data, key=parse_time)
        
        # Update bio_index mapping for sorted order
        sorted_nodes = []
        for i, profile in enumerate(sorted_profiles):
            # Find corresponding node
            original_id = f"person_{profile.get('bio_index', 0)}"
            node = next((n for n in self.nodes if n['id'] == original_id), None)
            if node:
                # Update ID to reflect sorted position
                node = node.copy()
                node['id'] = f"person_{i}"
                sorted_nodes.append(node)
        
        logger.info(f"[SORT] Sorted {len(sorted_nodes)} nodes by confirm_time")
        self.nodes = sorted_nodes
        return sorted_nodes
    
    def watts_strogatz_links(self, k: int = 6, beta: float = 0.618) -> List[Dict[str, Any]]:
        """Generate links using Watts-Strogatz small-world model."""
        logger.info(f"[LINKS] Generating Watts-Strogatz links (k={k}, beta={beta})")
        
        n = len(self.nodes)
        if n < k + 1:
            logger.warning(f"[LINKS] Not enough nodes ({n}) for k={k}, reducing k")
            k = max(2, n - 1)
        
        # Create Watts-Strogatz graph
        G = nx.watts_strogatz_graph(n, k, beta, seed=42)
        
        links = []
        
        for edge in G.edges():
            source_idx, target_idx = edge
            
            # Calculate similarity weight based on 768-dimensional embeddings
            source_vector = np.array(self.nodes[source_idx]['skill_vector'])
            target_vector = np.array(self.nodes[target_idx]['skill_vector'])
            
            # Cosine similarity for high-dimensional embeddings
            dot_product = np.dot(source_vector, target_vector)
            magnitude_source = np.linalg.norm(source_vector)
            magnitude_target = np.linalg.norm(target_vector)
            
            if magnitude_source > 0 and magnitude_target > 0:
                similarity = dot_product / (magnitude_source * magnitude_target)
                # Scale similarity to [0.1, 1.0] range for better link weights
                weight = max(0.1, min(1.0, (similarity + 1) / 2))  # Maps [-1,1] to [0.1,1.0]
            else:
                weight = 0.1
            
            link = {
                "source": self.nodes[source_idx]['id'],
                "target": self.nodes[target_idx]['id'],
                "weight": round(weight, 3)
            }
            
            links.append(link)
        
        logger.info(f"[LINKS] Generated {len(links)} links using Watts-Strogatz model")
        self.links = links
        return links
    
    def create_graph_format(self) -> Dict[str, Any]:
        """Create the final graph format."""
        logger.info("[GRAPH] Creating final graph format")
        
        graph = {
            "skill_lexicon": self.skill_lexicon,
            "nodes": self.nodes,
            "links": self.links
        }
        
        logger.info(f"[GRAPH] Graph created with {len(self.nodes)} nodes and {len(self.links)} links")
        return graph
    
    def save_graph(self, output_path: Path, graph: Dict[str, Any]):
        """Save the graph to JSON file."""
        logger.info(f"[SAVE] Saving graph to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, indent=2, ensure_ascii=False)
            logger.info("[SAVE] Graph saved successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save graph: {e}")


def main():
    """Main execution function."""
    
    logger.info("[START] Starting HTW Graph Formatter")
    
    project_dir = Path(__file__).parent
    input_path = project_dir / "local_skills_embeddings.json"
    output_path = project_dir / "htw_network_graph.json"
    
    if not input_path.exists():
        logger.error(f"[ERROR] Input file not found: {input_path}")
        return
    
    formatter = GraphFormatter()
    
    try:
        # Load embeddings data
        if not formatter.load_embeddings_data(input_path):
            return
        
        # Extract skill lexicon
        formatter.extract_skill_lexicon()
        
        # Create nodes
        formatter.create_nodes()
        
        # Sort by confirm_time for lattice
        formatter.sort_by_confirm_time()
        
        # Generate Watts-Strogatz links
        formatter.watts_strogatz_links(k=6, beta=0.618)
        
        # Create final graph format
        graph = formatter.create_graph_format()
        
        # Save graph
        formatter.save_graph(output_path, graph)
        
        logger.info(f"[COMPLETE] HTW Network Graph generation complete!")
        logger.info(f"[OUTPUT] Graph saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"[ERROR] Processing failed: {e}")
        raise e


if __name__ == "__main__":
    main()
