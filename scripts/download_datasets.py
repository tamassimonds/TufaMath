#!/usr/bin/env python3
"""
Dataset downloader that pre-caches datasets to avoid rate limits during training.
Run this before training to download and cache all datasets locally.
"""

import time
import sys
from pathlib import Path
import logging
from datasets import load_dataset

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from create_dataset import CORPORA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dataset_cached(hf_id, config=None, split="train"):
    """Check if dataset is already cached locally"""
    try:
        if config:
            ds = load_dataset(hf_id, config, split=split, streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(hf_id, split=split, streaming=True, trust_remote_code=True)
        
        # Try to get first example quickly
        next(iter(ds))
        return True
    except:
        return False

def download_with_retry(hf_id, config=None, split="train", max_retries=3, delay=30):
    """Download dataset with exponential backoff retry logic - PROPERLY CACHE NON-STREAMING"""
    
    # Check if already cached by trying to load non-streaming
    try:
        if config:
            ds_test = load_dataset(hf_id, config, split=split, trust_remote_code=True)
        else:
            ds_test = load_dataset(hf_id, split=split, trust_remote_code=True)
        logger.info(f"✓ {hf_id} already properly cached")
        return True
    except:
        logger.info(f"Need to download and cache {hf_id}")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading and caching {hf_id} (attempt {attempt + 1}/{max_retries})")
            
            # Download NON-STREAMING to properly cache
            if config:
                ds = load_dataset(
                    hf_id, 
                    config,
                    split=split, 
                    streaming=False,  # KEY CHANGE: Non-streaming for proper caching
                    trust_remote_code=True
                )
            else:
                ds = load_dataset(
                    hf_id,
                    split=split, 
                    streaming=False,  # KEY CHANGE: Non-streaming for proper caching
                    trust_remote_code=True
                )
            
            # Test access to first few examples to ensure cache is complete
            logger.info(f"  Verifying cache completeness for {hf_id}...")
            count = 0
            for example in ds:
                count += 1
                if count >= 100:  # Test more examples to ensure proper caching
                    break
            
            logger.info(f"  ✓ Successfully downloaded and cached {hf_id} ({count} examples verified)")
            return True
            
        except Exception as e:
            logger.warning(f"  ✗ Failed to download {hf_id}: {e}")
            
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"  Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                logger.error(f"  Failed to download {hf_id} after {max_retries} attempts")
                return False
            else:
                logger.info(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
    
    return False

def main():
    """Download and cache all datasets from CORPORA"""
    logger.info("Starting dataset download and caching process")
    logger.info(f"Will download {len(CORPORA)} datasets")
    
    successful = 0
    failed = []
    
    for i, spec in enumerate(CORPORA, 1):
        hf_id = spec["hf_id"]
        config = spec.get("config")
        split = spec["split"]
        
        logger.info(f"\n[{i}/{len(CORPORA)}] Processing {hf_id}")
        
        if download_with_retry(hf_id, config, split):
            successful += 1
            logger.info(f"✓ Completed {hf_id}")
        else:
            failed.append(hf_id)
            logger.error(f"✗ Failed {hf_id}")
        
        # Small delay between datasets to be nice to HF servers
        if i < len(CORPORA):
            logger.info("Waiting 1 seconds before next dataset...")
            time.sleep(1)
    
    # Summary
    logger.info(f"\n" + "="*60)
    logger.info(f"DOWNLOAD SUMMARY")
    logger.info(f"="*60)
    logger.info(f"Successfully cached: {successful}/{len(CORPORA)} datasets")
    
    if failed:
        logger.info(f"Failed datasets:")
        for dataset in failed:
            logger.info(f"  - {dataset}")
        logger.info(f"\nYou can retry failed datasets by running this script again.")
    else:
        logger.info(f"🎉 All datasets successfully cached!")
        logger.info(f"You can now run training without rate limit issues.")
    
    logger.info(f"="*60)

if __name__ == "__main__":
    main()