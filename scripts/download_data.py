#!/usr/bin/env python
"""
Download datasets for CausalShapGNN
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.downloader import DataDownloader


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for CausalShapGNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py --dataset gowalla
  python scripts/download_data.py --dataset all
  python scripts/download_data.py --list
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='all',
        choices=['gowalla', 'yelp2018', 'amazon-book', 'alibaba-ifashion',
                 'movielens-10m', 'movielens-1m', 'movielens-100k', 'all'],
        help='Dataset to download'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory to store datasets'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and their status'
    )
    
    args = parser.parse_args()
    
    downloader = DataDownloader(args.data_dir)
    
    if args.list:
        downloader.list_datasets()
    else:
        print(f"\n{'='*60}")
        print("CausalShapGNN Dataset Downloader")
        print(f"{'='*60}\n")
        
        downloader.download(args.dataset)
        
        print(f"\n{'='*60}")
        print("Download Complete!")
        print(f"{'='*60}\n")
        
        downloader.list_datasets()


if __name__ == "__main__":
    main()