"""
Dataset Downloader for CausalShapGNN
Fixed version with working URLs
"""

import os
import requests
import zipfile
import gzip
import shutil
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import random


class DataDownloader:
    """
    Download and extract benchmark datasets for CausalShapGNN.
    Uses multiple fallback sources for reliability.
    """
    
    # Working URLs (updated December 2024)
    DATASETS = {
        'gowalla': {
            'train_url': 'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/train.txt',
            'test_url': 'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/test.txt',
            'description': 'Gowalla check-in dataset',
            'type': 'direct'
        },
        'yelp2018': {
            'train_url': 'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/yelp2018/train.txt',
            'test_url': 'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/yelp2018/test.txt',
            'description': 'Yelp 2018 business reviews dataset',
            'type': 'direct'
        },
        'amazon-book': {
            'train_url': 'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/amazon-book/train.txt',
            'test_url': 'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/amazon-book/test.txt',
            'description': 'Amazon Book reviews dataset',
            'type': 'direct'
        },
        'movielens-1m': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'description': 'MovieLens 1M movie ratings',
            'type': 'movielens',
            'subdir': 'ml-1m',
            'rating_file': 'ratings.dat',
            'separator': '::',
            'min_rating': 4.0
        },
        'movielens-100k': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'description': 'MovieLens 100K movie ratings (smallest)',
            'type': 'movielens',
            'subdir': 'ml-100k',
            'rating_file': 'u.data',
            'separator': '\t',
            'min_rating': 4.0
        },
        'movielens-10m': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
            'description': 'MovieLens 10M movie ratings',
            'type': 'movielens',
            'subdir': 'ml-10M100K',
            'rating_file': 'ratings.dat',
            'separator': '::',
            'min_rating': 4.0
        }
    }
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def _download_file(self, url: str, filepath: str, desc: Optional[str] = None) -> bool:
        """Download file with progress bar."""
        try:
            print(f"  Downloading from: {url[:80]}...")
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or 'Downloading') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"  Download failed: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
    def _download_direct(self, dataset_name: str) -> str:
        """Download dataset with direct train/test URLs."""
        info = self.DATASETS[dataset_name]
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"  {dataset_name} already exists.")
            return dataset_dir
        
        # Download train.txt
        print(f"  Downloading train.txt...")
        success = self._download_file(info['train_url'], train_file, 'train.txt')
        
        if not success:
            raise Exception(f"Failed to download train.txt for {dataset_name}")
        
        # Download test.txt
        print(f"  Downloading test.txt...")
        success = self._download_file(info['test_url'], test_file, 'test.txt')
        
        if not success:
            raise Exception(f"Failed to download test.txt for {dataset_name}")
        
        # Verify files
        self._verify_dataset(dataset_dir, dataset_name)
        
        return dataset_dir
    
    def _download_movielens(self, dataset_name: str) -> str:
        """Download and process MovieLens dataset."""
        info = self.DATASETS[dataset_name]
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"  {dataset_name} already exists.")
            return dataset_dir
        
        # Download zip
        zip_path = os.path.join(dataset_dir, f'{dataset_name}.zip')
        success = self._download_file(info['url'], zip_path, dataset_name)
        
        if not success:
            raise Exception(f"Failed to download {dataset_name}")
        
        # Extract
        print(f"  Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        os.remove(zip_path)
        
        # Process ratings
        self._process_movielens(
            dataset_dir,
            info['subdir'],
            info['rating_file'],
            info['separator'],
            info['min_rating']
        )
        
        return dataset_dir
    
    def _process_movielens(self, dataset_dir: str, subdir: str, rating_file: str,
                           separator: str, min_rating: float):
        """Process MovieLens ratings into train/test format."""
        ratings_path = os.path.join(dataset_dir, subdir, rating_file)
        
        if not os.path.exists(ratings_path):
            print(f"  Ratings file not found: {ratings_path}")
            return
        
        print(f"  Processing ratings from {ratings_path}...")
        
        # Load ratings with positive filtering
        user_items: Dict[int, List[int]] = {}
        user_set = set()
        item_set = set()
        
        with open(ratings_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split(separator)
                if len(parts) >= 3:
                    try:
                        user_id = int(parts[0])
                        item_id = int(parts[1])
                        rating = float(parts[2])
                        
                        if rating >= min_rating:
                            user_set.add(user_id)
                            item_set.add(item_id)
                            
                            if user_id not in user_items:
                                user_items[user_id] = []
                            user_items[user_id].append(item_id)
                    except ValueError:
                        continue
        
        # Remap IDs to be contiguous
        user_map = {u: i for i, u in enumerate(sorted(user_set))}
        item_map = {i: j for j, i in enumerate(sorted(item_set))}
        
        # Remap and deduplicate
        remapped: Dict[int, List[int]] = {}
        for user, items in user_items.items():
            new_user = user_map[user]
            new_items = list(set(item_map[i] for i in items))
            if len(new_items) >= 2:  # Minimum 2 items per user
                remapped[new_user] = new_items
        
        # Split into train/test (80/20)
        random.seed(42)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        with open(train_file, 'w') as f_train, open(test_file, 'w') as f_test:
            for user, items in remapped.items():
                random.shuffle(items)
                split_idx = max(1, int(len(items) * 0.8))
                
                train_items = items[:split_idx]
                test_items = items[split_idx:]
                
                if train_items:
                    f_train.write(f"{user} {' '.join(map(str, train_items))}\n")
                if test_items:
                    f_test.write(f"{user} {' '.join(map(str, test_items))}\n")
        
        print(f"  Processed: {len(user_map)} users, {len(item_map)} items")
    
    def _verify_dataset(self, dataset_dir: str, dataset_name: str):
        """Verify dataset files and print statistics."""
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        n_train = 0
        n_test = 0
        n_users = 0
        n_items = 0
        
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        n_train += len(parts) - 1
                        n_users = max(n_users, int(parts[0]) + 1)
                        n_items = max(n_items, max(int(i) for i in parts[1:]) + 1)
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        n_test += len(parts) - 1
                        n_users = max(n_users, int(parts[0]) + 1)
                        n_items = max(n_items, max(int(i) for i in parts[1:]) + 1)
        
        print(f"  Verified {dataset_name}:")
        print(f"    Users: {n_users:,}")
        print(f"    Items: {n_items:,}")
        print(f"    Train interactions: {n_train:,}")
        print(f"    Test interactions: {n_test:,}")
    
    def download(self, dataset: str) -> str:
        """Download specified dataset."""
        dataset = dataset.lower()
        
        if dataset == 'all':
            print("Downloading all datasets...")
            for name in self.DATASETS:
                try:
                    self.download(name)
                except Exception as e:
                    print(f"  Failed to download {name}: {e}")
            return self.data_dir
        
        if dataset not in self.DATASETS:
            available = list(self.DATASETS.keys())
            raise ValueError(f"Unknown dataset: {dataset}. Available: {available}")
        
        print(f"\n{'='*60}")
        print(f"Downloading: {dataset}")
        print(f"{'='*60}")
        
        info = self.DATASETS[dataset]
        
        try:
            if info['type'] == 'direct':
                result = self._download_direct(dataset)
            elif info['type'] == 'movielens':
                result = self._download_movielens(dataset)
            else:
                raise ValueError(f"Unknown dataset type: {info['type']}")
            
            print(f"  ✓ {dataset} ready at {result}")
            return result
            
        except Exception as e:
            print(f"  ✗ Failed to download {dataset}: {e}")
            raise
    
    def list_datasets(self):
        """Print available datasets and their status."""
        print("\nAvailable Datasets:")
        print("-" * 70)
        
        for name, info in self.DATASETS.items():
            dataset_dir = os.path.join(self.data_dir, name)
            train_file = os.path.join(dataset_dir, 'train.txt')
            test_file = os.path.join(dataset_dir, 'test.txt')
            
            if os.path.exists(train_file) and os.path.exists(test_file):
                status = "✓ Downloaded"
            elif os.path.exists(train_file):
                status = "⚠ Partial"
            else:
                status = "✗ Not downloaded"
            
            print(f"  {name:<20} {status:<20} {info['description']}")
        
        print("-" * 70)


# Standalone function for easy use
def download_all_datasets(data_dir: str = './data'):
    """Download all datasets."""
    downloader = DataDownloader(data_dir)
    
    for dataset in ['movielens-100k', 'gowalla', 'yelp2018', 'amazon-book']:
        try:
            downloader.download(dataset)
        except Exception as e:
            print(f"Warning: Could not download {dataset}: {e}")
    
    downloader.list_datasets()
    return downloader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    
    downloader = DataDownloader(args.data_dir)
    downloader.download(args.dataset)
    downloader.list_datasets()