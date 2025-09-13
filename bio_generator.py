#!/usr/bin/env python3
"""
HTW Network Bio Generator
Generates simulated JSON bios with skill sets and experience years based on job titles and industries.
"""

import os
import json
import csv
import asyncio
import logging
import time
import signal
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from pydantic import BaseModel, Field

# Configure logging with Windows-safe encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bio_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SkillSet(BaseModel):
    """Individual skill with experience years."""
    skill: str = Field(description="Name of the skill")
    years: int = Field(description="Years of experience with this skill", ge=0, le=50)
    proficiency: str = Field(description="Proficiency level", pattern="^(Beginner|Intermediate|Advanced|Expert)$")


class PersonBio(BaseModel):
    """Complete person bio with skills and metadata."""
    role: str = Field(description="Job title/role")
    industry: str = Field(description="Industry sector")
    location: str = Field(description="Geographic location")
    skills: List[SkillSet] = Field(description="List of skills with experience")
    total_experience_years: int = Field(description="Total years in field", ge=0, le=50)
    education_level: str = Field(description="Highest education level")
    specializations: List[str] = Field(description="Key areas of specialization")


class BioGenerator:
    """Generates simulated bios using Claude API."""
    
    def __init__(self, api_key: str):
        logger.info("[INIT] Initializing BioGenerator with Claude API")
        self.client = Anthropic(api_key=api_key)
        self.generated_bios: List[PersonBio] = []
        self.interrupt_requested = False
        logger.info("[INIT] BioGenerator initialized successfully")
    
    def save_progress(self, results: List[Dict[str, Any]], output_path: Path):
        """Save current progress to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[SAVE] Progress saved: {len(results)} bios")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save progress: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.info("[INTERRUPT] Received interrupt signal, finishing current bio...")
        self.interrupt_requested = True
    
    def load_existing_bios(self, output_path: Path) -> List[Dict[str, Any]]:
        """Load existing generated bios from file if it exists."""
        if not output_path.exists():
            logger.info("[RESUME] No existing output file found, starting fresh")
            return []
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_bios = json.load(f)
            logger.info(f"[RESUME] Loaded {len(existing_bios)} existing bios from {output_path}")
            return existing_bios
        except Exception as e:
            logger.error(f"[ERROR] Failed to load existing bios: {e}")
            return []
    
    def create_processed_set(self, existing_bios: List[Dict[str, Any]]) -> set:
        """Create a set of (role, industry) tuples for already processed entries."""
        processed = set()
        for bio in existing_bios:
            role = bio.get('role', '').strip().lower()
            industry = bio.get('industry', '').strip().lower()
            if role and industry:
                processed.add((role, industry))
        logger.info(f"[RESUME] Created processed set with {len(processed)} unique role/industry combinations")
        return processed
    
    def generate_bio(self, role: str, industry: str, location: str = "") -> Optional[PersonBio]:
        """Generate a simulated bio for a person based on role and industry."""
        
        logger.info(f"[GEN] Starting bio generation for: '{role}' in '{industry}' at '{location}'")
        start_time = time.time()
        
        prompt = f"""
        Create a realistic professional bio simulation for someone with this profile:
        - Role: {role}
        - Industry: {industry}
        - Location: {location}
        
        Generate a JSON response with the following structure:
        {{
            "role": "{role}",
            "industry": "{industry}",
            "location": "{location}",
            "skills": [
                {{
                    "skill": "skill name",
                    "years": number_of_years,
                    "proficiency": "Beginner|Intermediate|Advanced|Expert"
                }}
            ],
            "total_experience_years": number,
            "education_level": "High School|Associates|Bachelors|Masters|PhD|Professional",
            "specializations": ["specialization1", "specialization2"]
        }}
        
        Requirements:
        - Include 5-8 relevant skills for this role/industry
        - Years should be realistic for career progression
        - Proficiency should match years of experience
        - Specializations should be industry-specific
        - Make it diverse and realistic
        
        Return ONLY the JSON, no additional text.
        """
        
        try:
            logger.info(f"[API] Sending request to Claude API for '{role}'")
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[API] Received response from Claude API in {elapsed:.2f}s")
            
            # Log the raw response for debugging
            raw_response = response.content[0].text.strip()
            logger.debug(f"Raw Claude response: {raw_response[:200]}...")
            
            # Parse the JSON response
            logger.info(f"[PARSE] Parsing JSON response for '{role}'")
            bio_data = json.loads(raw_response)
            bio = PersonBio(**bio_data)
            
            logger.info(f"[SUCCESS] Generated bio for '{role}' with {len(bio.skills)} skills")
            return bio
            
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] JSON parsing failed for '{role}': {e}")
            logger.error(f"Raw response was: {raw_response}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Error generating bio for '{role}' in '{industry}': {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return None
    
    def process_csv(self, csv_path: Path, output_path: Path, max_rows: Optional[int] = None, resume: bool = True):
        """Process CSV file and generate bios for each row with resume capability."""
        
        logger.info(f"[CSV] Starting CSV processing from: {csv_path}")
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Load existing bios for resume capability
        existing_bios = []
        processed_set = set()
        if resume:
            existing_bios = self.load_existing_bios(output_path)
            processed_set = self.create_processed_set(existing_bios)
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"[CSV] Successfully loaded CSV with {len(df)} total rows")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load CSV: {e}")
            return []
        
        if max_rows:
            df = df.head(max_rows)
            logger.info(f"[CSV] Limited processing to first {max_rows} rows")
        
        logger.info(f"[CSV] CSV columns: {list(df.columns)}")
        
        # Start with existing bios if resuming
        results = existing_bios.copy()
        processed_count = len(existing_bios)
        skipped_count = 0
        resume_skipped = 0
        
        for idx, row in df.iterrows():
            role = str(row.get('Role / Job Title', '')).strip()
            industry = str(row.get('Industry', '')).strip()
            
            logger.debug(f"[ROW] Row {idx+1}: role='{role}', industry='{industry}'")
            
            # Skip empty or invalid rows
            if not role or role == '-' or not industry or industry == 'nan':
                logger.warning(f"[SKIP] Skipping row {idx+1}: empty role or industry")
                skipped_count += 1
                continue
            
            # Check if already processed (resume capability)
            role_key = role.lower()
            industry_key = industry.lower()
            if resume and (role_key, industry_key) in processed_set:
                logger.info(f"[RESUME] Skipping row {idx+1}: already processed '{role}' in '{industry}'")
                resume_skipped += 1
                continue
            
            # Create location string
            city = str(row.get('City', '')).strip()
            state = str(row.get('State / Province', '')).strip()
            country = str(row.get('Country', '')).strip()
            location_parts = [p for p in [city, state, country] if p and p != 'nan']
            location = ', '.join(location_parts)
            
            logger.info(f"[PROC] Processing [{idx+1}/{len(df)}]: '{role}' in '{industry}' | {location}")
            
            # Generate bio using Claude
            bio = self.generate_bio(role, industry, location)
            
            if bio:
                results.append(bio.model_dump())
                processed_count += 1
                logger.info(f"[SUCCESS] [{processed_count}] Successfully generated bio for '{role}'")
                
                # Add to processed set for future resume operations
                processed_set.add((role_key, industry_key))
                
                # Save progress every 10 bios
                if processed_count % 10 == 0:
                    self.save_progress(results, output_path)
                
                # Add a small delay to be respectful to API
                time.sleep(0.5)
            else:
                logger.error(f"[ERROR] Failed to generate bio for '{role}' in '{industry}'")
            
            # Check for interrupt request
            if self.interrupt_requested:
                logger.info("[INTERRUPT] Saving progress and exiting gracefully...")
                self.save_progress(results, output_path)
                logger.info(f"[INTERRUPT] Saved {len(results)} bios before exit")
                return results
        
        # Save results
        logger.info(f"[SAVE] Saving {len(results)} bios to: {output_path}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[SAVE] Successfully saved {len(results)} bios")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save results: {e}")
            return results
        
        logger.info(f"[COMPLETE] Processing complete! Generated: {processed_count}, Skipped: {skipped_count}, Resume skipped: {resume_skipped}")
        return results


def main():
    """Main execution function."""
    
    logger.info("[START] Starting HTW Bio Generator")
    start_time = datetime.now()
    
    # Load environment variables
    logger.info("[ENV] Loading environment variables")
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        logger.error("[ERROR] ANTHROPIC_API_KEY not found in environment variables")
        logger.error("Please ensure your .env file contains: ANTHROPIC_API_KEY=your_key_here")
        return
    
    logger.info("[ENV] Found Anthropic API key")
    
    # Set up paths
    project_dir = Path(__file__).parent
    csv_path = project_dir / "HTW Network Data - Pseunom Subset of ~2.5K People - HTW2025Audience.csv"
    output_path = project_dir / "generated_bios.json"
    
    logger.info(f"[PATH] Project directory: {project_dir}")
    logger.info(f"[PATH] CSV path: {csv_path}")
    logger.info(f"[PATH] Output path: {output_path}")
    
    if not csv_path.exists():
        logger.error(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    # Initialize generator
    generator = BioGenerator(api_key)
    
    # Process CSV (full dataset - remove max_rows limit)
    logger.info("[START] Starting bio generation process...")
    results = generator.process_csv(csv_path, output_path, max_rows=None)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"[COMPLETE] Generated {len(results)} professional bios")
    logger.info(f"[OUTPUT] Output saved to: {output_path}")
    logger.info(f"[TIME] Total execution time: {duration}")
    logger.info(f"[LOG] Log file saved to: bio_generation.log")


if __name__ == "__main__":
    main()
