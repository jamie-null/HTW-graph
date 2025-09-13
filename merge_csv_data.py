#!/usr/bin/env python3
"""
CSV Data Merger
Merges additional CSV columns (dates, location) with generated bio JSON.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_csv_with_bios(csv_path: Path, json_path: Path, output_path: Path):
    """Merge CSV columns with generated bios JSON based on sequential order."""
    
    logger.info(f"[MERGE] Loading CSV from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"[MERGE] Loaded CSV with {len(df)} rows")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load CSV: {e}")
        return
    
    logger.info(f"[MERGE] Loading JSON from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            bios = json.load(f)
        logger.info(f"[MERGE] Loaded {len(bios)} bios from JSON")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load JSON: {e}")
        return
    
    logger.info(f"[MERGE] CSV columns: {list(df.columns)}")
    
    # Track which CSV rows have been processed
    csv_processed_count = 0
    merged_bios = []
    
    for bio_idx, bio in enumerate(bios):
        # Find corresponding CSV row by matching role and industry
        found_match = False
        
        for csv_idx in range(csv_processed_count, len(df)):
            row = df.iloc[csv_idx]
            csv_role = str(row.get('Role / Job Title', '')).strip()
            csv_industry = str(row.get('Industry', '')).strip()
            
            # Skip empty rows in CSV
            if not csv_role or csv_role == '-' or not csv_industry or csv_industry == 'nan':
                csv_processed_count += 1
                continue
            
            # Check if this matches the bio
            if (bio['role'].lower() == csv_role.lower() and 
                bio['industry'].lower() == csv_industry.lower()):
                
                # Merge CSV data into bio
                enhanced_bio = bio.copy()
                enhanced_bio.update({
                    'csv_row_index': csv_idx + 1,
                    'confirm_time': str(row.get('CONFIRM_TIME', '')),
                    'last_changed': str(row.get('LAST_CHANGED', '')),
                    'city': str(row.get('City', '')).strip(),
                    'state_province': str(row.get('State / Province', '')).strip(),
                    'country': str(row.get('Country', '')).strip()
                })
                
                merged_bios.append(enhanced_bio)
                csv_processed_count = csv_idx + 1
                found_match = True
                logger.info(f"[MATCH] Bio {bio_idx+1} matched to CSV row {csv_idx+1}: '{csv_role}'")
                break
        
        if not found_match:
            logger.warning(f"[NO_MATCH] Bio {bio_idx+1} ('{bio['role']}') not matched to CSV")
            merged_bios.append(bio)
    
    # Save merged results
    logger.info(f"[SAVE] Saving {len(merged_bios)} merged bios to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_bios, f, indent=2, ensure_ascii=False)
        logger.info(f"[SAVE] Successfully saved merged data")
    except Exception as e:
        logger.error(f"[ERROR] Failed to save merged data: {e}")
        return
    
    logger.info(f"[COMPLETE] Merge complete! Enhanced {len(merged_bios)} bios with CSV data")


def main():
    """Main execution function."""
    
    project_dir = Path(__file__).parent
    csv_path = project_dir / "HTW Network Data - Pseunom Subset of ~2.5K People - HTW2025Audience.csv"
    json_path = project_dir / "generated_bios.json"
    output_path = project_dir / "enhanced_bios.json"
    
    logger.info("[START] Starting CSV-Bio merge process")
    
    if not csv_path.exists():
        logger.error(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    if not json_path.exists():
        logger.error(f"[ERROR] JSON file not found: {json_path}")
        return
    
    merge_csv_with_bios(csv_path, json_path, output_path)
    
    logger.info(f"[COMPLETE] Merge process complete!")
    logger.info(f"[OUTPUT] Enhanced bios saved to: {output_path}")


if __name__ == "__main__":
    main()
