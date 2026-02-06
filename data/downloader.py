"""
Dataset Downloader for CausalShapGNN
Supports: Gowalla, Yelp2018, Amazon-Book, Alibaba-iFashion, MovieLens-10M
"""

import os
import requests
import zipfile
import gzip
import shutil
import pandas as pd
from tqdm import tqdm
from typing import Optional
import hashlib


class DataDownloader:
    """
    Download and extract benchmark datasets for CausalShapGNN.
    """
    
    # Dataset URLs and metadata
    DATASETS = {
        'gowalla': {
            'url': 'https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz',
            'alt_url': 'https://github.com/kuandeng/LightGCN/raw/master/Data/gowalla.zip',
            'gdrive_id': '1seZIaIR8_E2R9U1EboltHv5woGUaFxoq',
            'description': 'Gowalla check-in dataset',
            'format': 'txt',
            'md5': None
        },
        'yelp2018': {
            'url': 'https://github.com/kuandeng/LightGCN/raw/master/Data/yelp2018.zip',
            'gdrive_id': '1yMxIwKQ3ctMJ1qE4I_GpbpLiKWdNObb6',
            'description': 'Yelp 2018 business reviews dataset',
            'format': 'zip',
            'md5': None
        },
        'amazon-book': {
            'url': 'https://github.com/kuandeng/LightGCN/raw/master/Data/amazon-book.zip',
            'gdrive_id': '1FTd4I0o8K7T7zYpgkF7ZH7Mw2GhKOfrp',
            'description': 'Amazon Book reviews dataset',
            'format': 'zip',
            'md5': None
        },
        'alibaba-ifashion': {
            'url': 'https://github.com/wenyuer/POG/raw/master/data/ifashion.zip',
            'gdrive_id': '1f0QXFSnwCsrMoaqy_DTxA_YJg1l2UTKY',
            'description': 'Alibaba iFashion outfit dataset',
            'format': 'zip',
            'md5': None
        },
        'movielens-10m': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
            'description': 'MovieLens 10M movie ratings',
            'format': 'zip',
            'md5': 'ce571fd55effeba0271552578f2648bd'
        },
        'movielens-1m': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'description': 'MovieLens 1M movie ratings (smaller)',
            'format': 'zip',
            'md5': 'c4d9eecfca2ab87c1945afe126590906'
        },
        'movielens-100k': {
            'url': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'description': 'MovieLens 100K movie ratings (smallest)',
            'format': 'zip',
            'md5': '0e33842e24a9c977be4e0107933c0723'
        }
    }
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize downloader.
        
        Args:
            data_dir: Root directory for storing datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_file(self, url: str, filepath: str, 
                      desc: Optional[str] = None) -> bool:
        """
        Download file with progress bar.
        
        Args:
            url: URL to download from
            filepath: Local path to save file
            desc: Description for progress bar
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=desc or 'Downloading') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
    def download_from_gdrive(self, file_id: str, filepath: str,
                             desc: Optional[str] = None) -> bool:
        """
        Download file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            filepath: Local path to save file
            desc: Description for progress bar
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import gdown
            gdown.download(id=file_id, output=filepath, quiet=False)
            return os.path.exists(filepath)
        except ImportError:
            print("gdown not installed. Install with: pip install gdown")
            return False
        except Exception as e:
            print(f"Google Drive download failed: {e}")
            return False
    
    def extract_zip(self, zip_path: str, extract_dir: str):
        """Extract ZIP file"""
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
    
    def extract_gzip(self, gz_path: str, output_path: str):
        """Extract GZIP file"""
        print(f"Extracting {gz_path}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted to {output_path}")
    
    def verify_md5(self, filepath: str, expected_md5: str) -> bool:
        """Verify file MD5 checksum"""
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest() == expected_md5
    
    def download_gowalla(self) -> str:
        """
        Download Gowalla dataset.
        
        Returns:
            Path to dataset directory
        """
        dataset_dir = os.path.join(self.data_dir, 'gowalla')
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("Gowalla dataset already exists.")
            return dataset_dir
        
        print("Downloading Gowalla dataset...")
        
        # Try primary URL (LightGCN processed data)
        zip_path = os.path.join(dataset_dir, 'gowalla.zip')
        
        success = self.download_file(
            self.DATASETS['gowalla']['alt_url'],
            zip_path,
            'Gowalla'
        )
        
        if not success:
            # Try Google Drive
            print("Trying Google Drive...")
            success = self.download_from_gdrive(
                self.DATASETS['gowalla']['gdrive_id'],
                zip_path,
                'Gowalla'
            )
        
        if success and os.path.exists(zip_path):
            self.extract_zip(zip_path, dataset_dir)
            os.remove(zip_path)
            
            # Move files from subdirectory if needed
            subdir = os.path.join(dataset_dir, 'gowalla')
            if os.path.isdir(subdir):
                for f in os.listdir(subdir):
                    shutil.move(os.path.join(subdir, f), dataset_dir)
                os.rmdir(subdir)
        
        if not os.path.exists(train_file):
            print("Downloading raw Gowalla and processing...")
            self._download_and_process_raw_gowalla(dataset_dir)
        
        print(f"Gowalla dataset ready at {dataset_dir}")
        return dataset_dir
    
    def _download_and_process_raw_gowalla(self, dataset_dir: str):
        """Download raw Gowalla data and process it"""
        gz_path = os.path.join(dataset_dir, 'gowalla.txt.gz')
        txt_path = os.path.join(dataset_dir, 'gowalla_checkins.txt')
        
        # Download raw check-ins
        success = self.download_file(
            self.DATASETS['gowalla']['url'],
            gz_path,
            'Gowalla Raw'
        )
        
        if success:
            self.extract_gzip(gz_path, txt_path)
            os.remove(gz_path)
            
            # Process raw data
            self._process_gowalla_raw(txt_path, dataset_dir)
    
    def _process_gowalla_raw(self, raw_path: str, dataset_dir: str):
        """Process raw Gowalla check-ins into train/test splits"""
        print("Processing raw Gowalla data...")
        
        # Load check-ins
        user_items = {}
        with open(raw_path, 'r') as f:
            for line in tqdm(f, desc='Loading'):
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    user_id = parts[0]
                    location_id = parts[4]
                    
                    if user_id not in user_items:
                        user_items[user_id] = set()
                    user_items[user_id].add(location_id)
        
        # Filter users and items with minimum interactions
        min_interactions = 10
        
        # Count item frequencies
        item_counts = {}
        for items in user_items.values():
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        # Filter items
        valid_items = {i for i, c in item_counts.items() if c >= min_interactions}
        
        # Filter users and remap IDs
        user_map = {}
        item_map = {}
        
        filtered_data = {}
        for user, items in user_items.items():
            valid = [i for i in items if i in valid_items]
            if len(valid) >= min_interactions:
                if user not in user_map:
                    user_map[user] = len(user_map)
                
                user_id = user_map[user]
                filtered_data[user_id] = []
                
                for item in valid:
                    if item not in item_map:
                        item_map[item] = len(item_map)
                    filtered_data[user_id].append(item_map[item])
        
        # Split into train/test
        import random
        random.seed(42)
        
        train_data = {}
        test_data = {}
        
        for user, items in filtered_data.items():
            random.shuffle(items)
            split_idx = int(len(items) * 0.8)
            train_data[user] = items[:split_idx]
            test_data[user] = items[split_idx:]
        
        # Write files
        with open(os.path.join(dataset_dir, 'train.txt'), 'w') as f:
            for user, items in train_data.items():
                if items:
                    f.write(f"{user} {' '.join(map(str, items))}\n")
        
        with open(os.path.join(dataset_dir, 'test.txt'), 'w') as f:
            for user, items in test_data.items():
                if items:
                    f.write(f"{user} {' '.join(map(str, items))}\n")
        
        print(f"Processed: {len(user_map)} users, {len(item_map)} items")
    
    def download_yelp2018(self) -> str:
        """
        Download Yelp2018 dataset.
        
        Returns:
            Path to dataset directory
        """
        dataset_dir = os.path.join(self.data_dir, 'yelp2018')
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("Yelp2018 dataset already exists.")
            return dataset_dir
        
        print("Downloading Yelp2018 dataset...")
        
        zip_path = os.path.join(dataset_dir, 'yelp2018.zip')
        
        success = self.download_file(
            self.DATASETS['yelp2018']['url'],
            zip_path,
            'Yelp2018'
        )
        
        if not success:
            print("Trying Google Drive...")
            success = self.download_from_gdrive(
                self.DATASETS['yelp2018']['gdrive_id'],
                zip_path,
                'Yelp2018'
            )
        
        if success and os.path.exists(zip_path):
            self.extract_zip(zip_path, dataset_dir)
            os.remove(zip_path)
            
            # Handle nested directory
            subdir = os.path.join(dataset_dir, 'yelp2018')
            if os.path.isdir(subdir):
                for f in os.listdir(subdir):
                    src = os.path.join(subdir, f)
                    dst = os.path.join(dataset_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                shutil.rmtree(subdir, ignore_errors=True)
        
        print(f"Yelp2018 dataset ready at {dataset_dir}")
        return dataset_dir
    
    def download_amazon_book(self) -> str:
        """
        Download Amazon-Book dataset.
        
        Returns:
            Path to dataset directory
        """
        dataset_dir = os.path.join(self.data_dir, 'amazon-book')
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("Amazon-Book dataset already exists.")
            return dataset_dir
        
        print("Downloading Amazon-Book dataset...")
        
        zip_path = os.path.join(dataset_dir, 'amazon-book.zip')
        
        success = self.download_file(
            self.DATASETS['amazon-book']['url'],
            zip_path,
            'Amazon-Book'
        )
        
        if not success:
            print("Trying Google Drive...")
            success = self.download_from_gdrive(
                self.DATASETS['amazon-book']['gdrive_id'],
                zip_path,
                'Amazon-Book'
            )
        
        if success and os.path.exists(zip_path):
            self.extract_zip(zip_path, dataset_dir)
            os.remove(zip_path)
            
            # Handle nested directory
            subdir = os.path.join(dataset_dir, 'amazon-book')
            if os.path.isdir(subdir):
                for f in os.listdir(subdir):
                    src = os.path.join(subdir, f)
                    dst = os.path.join(dataset_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                shutil.rmtree(subdir, ignore_errors=True)
        
        print(f"Amazon-Book dataset ready at {dataset_dir}")
        return dataset_dir
    
    def download_alibaba_ifashion(self) -> str:
        """
        Download Alibaba-iFashion dataset.
        
        Returns:
            Path to dataset directory
        """
        dataset_dir = os.path.join(self.data_dir, 'alibaba-ifashion')
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("Alibaba-iFashion dataset already exists.")
            return dataset_dir
        
        print("Downloading Alibaba-iFashion dataset...")
        
        zip_path = os.path.join(dataset_dir, 'ifashion.zip')
        
        success = self.download_file(
            self.DATASETS['alibaba-ifashion']['url'],
            zip_path,
            'Alibaba-iFashion'
        )
        
        if not success:
            print("Trying Google Drive...")
            success = self.download_from_gdrive(
                self.DATASETS['alibaba-ifashion']['gdrive_id'],
                zip_path,
                'Alibaba-iFashion'
            )
        
        if success and os.path.exists(zip_path):
            self.extract_zip(zip_path, dataset_dir)
            os.remove(zip_path)
            
            # Process iFashion format if needed
            self._process_ifashion(dataset_dir)
        
        print(f"Alibaba-iFashion dataset ready at {dataset_dir}")
        return dataset_dir
    
    def _process_ifashion(self, dataset_dir: str):
        """Process iFashion data format"""
        # Check for raw files and convert if needed
        user_item_file = os.path.join(dataset_dir, 'user_item.txt')
        
        if os.path.exists(user_item_file) and not os.path.exists(
            os.path.join(dataset_dir, 'train.txt')):
            
            print("Processing iFashion data...")
            
            user_items = {}
            with open(user_item_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        user = int(parts[0])
                        item = int(parts[1])
                        if user not in user_items:
                            user_items[user] = []
                        user_items[user].append(item)
            
            # Split
            import random
            random.seed(42)
            
            with open(os.path.join(dataset_dir, 'train.txt'), 'w') as f_train, \
                 open(os.path.join(dataset_dir, 'test.txt'), 'w') as f_test:
                
                for user, items in user_items.items():
                    random.shuffle(items)
                    split_idx = int(len(items) * 0.8)
                    
                    train_items = items[:split_idx]
                    test_items = items[split_idx:]
                    
                    if train_items:
                        f_train.write(f"{user} {' '.join(map(str, train_items))}\n")
                    if test_items:
                        f_test.write(f"{user} {' '.join(map(str, test_items))}\n")
    
    def download_movielens_10m(self) -> str:
        """
        Download MovieLens-10M dataset.
        
        Returns:
            Path to dataset directory
        """
        dataset_dir = os.path.join(self.data_dir, 'movielens-10m')
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("MovieLens-10M dataset already exists.")
            return dataset_dir
        
        print("Downloading MovieLens-10M dataset...")
        
        zip_path = os.path.join(dataset_dir, 'ml-10m.zip')
        
        success = self.download_file(
            self.DATASETS['movielens-10m']['url'],
            zip_path,
            'MovieLens-10M'
        )
        
        if success and os.path.exists(zip_path):
            # Verify checksum
            if self.DATASETS['movielens-10m']['md5']:
                if not self.verify_md5(zip_path, self.DATASETS['movielens-10m']['md5']):
                    print("Warning: MD5 checksum mismatch")
            
            self.extract_zip(zip_path, dataset_dir)
            os.remove(zip_path)
            
            # Process MovieLens format
            self._process_movielens(dataset_dir, 'ml-10M100K')
        
        print(f"MovieLens-10M dataset ready at {dataset_dir}")
        return dataset_dir
    
    def download_movielens_1m(self) -> str:
        """Download MovieLens-1M dataset (smaller alternative)"""
        dataset_dir = os.path.join(self.data_dir, 'movielens-1m')
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("MovieLens-1M dataset already exists.")
            return dataset_dir
        
        print("Downloading MovieLens-1M dataset...")
        
        zip_path = os.path.join(dataset_dir, 'ml-1m.zip')
        
        success = self.download_file(
            self.DATASETS['movielens-1m']['url'],
            zip_path,
            'MovieLens-1M'
        )
        
        if success and os.path.exists(zip_path):
            self.extract_zip(zip_path, dataset_dir)
            os.remove(zip_path)
            self._process_movielens(dataset_dir, 'ml-1m')
        
        print(f"MovieLens-1M dataset ready at {dataset_dir}")
        return dataset_dir
    
    def download_movielens_100k(self) -> str:
        """Download MovieLens-100K dataset (smallest, for testing)"""
        dataset_dir = os.path.join(self.data_dir, 'movielens-100k')
        os.makedirs(dataset_dir, exist_ok=True)
        
        train_file = os.path.join(dataset_dir, 'train.txt')
        test_file = os.path.join(dataset_dir, 'test.txt')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("MovieLens-100K dataset already exists.")
            return dataset_dir
        
        print("Downloading MovieLens-100K dataset...")
        
        zip_path = os.path.join(dataset_dir, 'ml-100k.zip')
        
        success = self.download_file(
            self.DATASETS['movielens-100k']['url'],
            zip_path,
            'MovieLens-100K'
        )
        
        if success and os.path.exists(zip_path):
            self.extract_zip(zip_path, dataset_dir)
            os.remove(zip_path)
            self._process_movielens_100k(dataset_dir)
        
        print(f"MovieLens-100K dataset ready at {dataset_dir}")
        return dataset_dir
    
    def _process_movielens(self, dataset_dir: str, subdir_name: str):
        """Process MovieLens data format (1M/10M)"""
        subdir = os.path.join(dataset_dir, subdir_name)
        ratings_file = os.path.join(subdir, 'ratings.dat')
        
        if not os.path.exists(ratings_file):
            print(f"Ratings file not found: {ratings_file}")
            return
        
        print("Processing MovieLens data...")
        
        # Load ratings
        ratings = []
        with open(ratings_file, 'r', encoding='latin-1') as f:
            for line in tqdm(f, desc='Loading ratings'):
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    user_id = int(parts[0])
                    item_id = int(parts[1])
                    rating = float(parts[2])
                    
                    # Use implicit feedback (rating >= 4)
                    if rating >= 4.0:
                        ratings.append((user_id, item_id))
        
        # Remap IDs
        users = sorted(set(r[0] for r in ratings))
        items = sorted(set(r[1] for r in ratings))
        
        user_map = {u: i for i, u in enumerate(users)}
        item_map = {i: j for j, i in enumerate(items)}
        
        # Group by user
        user_items = {}
        for user, item in ratings:
            user_id = user_map[user]
            item_id = item_map[item]
            
            if user_id not in user_items:
                user_items[user_id] = []
            user_items[user_id].append(item_id)
        
        # Remove duplicates
        for user in user_items:
            user_items[user] = list(set(user_items[user]))
        
        # Split
        import random
        random.seed(42)
        
        with open(os.path.join(dataset_dir, 'train.txt'), 'w') as f_train, \
             open(os.path.join(dataset_dir, 'test.txt'), 'w') as f_test:
            
            for user, items in user_items.items():
                if len(items) < 2:
                    continue
                    
                random.shuffle(items)
                split_idx = max(1, int(len(items) * 0.8))
                
                train_items = items[:split_idx]
                test_items = items[split_idx:]
                
                if train_items:
                    f_train.write(f"{user} {' '.join(map(str, train_items))}\n")
                if test_items:
                    f_test.write(f"{user} {' '.join(map(str, test_items))}\n")
        
        print(f"Processed: {len(users)} users, {len(items)} items")
    
    def _process_movielens_100k(self, dataset_dir: str):
        """Process MovieLens-100K data format"""
        subdir = os.path.join(dataset_dir, 'ml-100k')
        data_file = os.path.join(subdir, 'u.data')
        
        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            return
        
        print("Processing MovieLens-100K data...")
        
        ratings = []
        with open(data_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    user_id = int(parts[0])
                    item_id = int(parts[1])
                    rating = float(parts[2])
                    
                    if rating >= 4.0:
                        ratings.append((user_id - 1, item_id - 1))  # 0-indexed
        
        # Group by user
        user_items = {}
        for user, item in ratings:
            if user not in user_items:
                user_items[user] = []
            user_items[user].append(item)
        
        # Remove duplicates
        for user in user_items:
            user_items[user] = list(set(user_items[user]))
        
        # Split
        import random
        random.seed(42)
        
        with open(os.path.join(dataset_dir, 'train.txt'), 'w') as f_train, \
             open(os.path.join(dataset_dir, 'test.txt'), 'w') as f_test:
            
            for user, items in user_items.items():
                if len(items) < 2:
                    continue
                    
                random.shuffle(items)
                split_idx = max(1, int(len(items) * 0.8))
                
                train_items = items[:split_idx]
                test_items = items[split_idx:]
                
                if train_items:
                    f_train.write(f"{user} {' '.join(map(str, train_items))}\n")
                if test_items:
                    f_test.write(f"{user} {' '.join(map(str, test_items))}\n")
    
    def download(self, dataset: str) -> str:
        """
        Download specified dataset.
        
        Args:
            dataset: Dataset name ('gowalla', 'yelp2018', 'amazon-book', 
                    'alibaba-ifashion', 'movielens-10m', 'movielens-1m', 
                    'movielens-100k', or 'all')
        
        Returns:
            Path to dataset directory (or data root for 'all')
        """
        dataset = dataset.lower()
        
        if dataset == 'all':
            print("Downloading all datasets...")
            self.download_gowalla()
            self.download_yelp2018()
            self.download_amazon_book()
            self.download_alibaba_ifashion()
            self.download_movielens_10m()
            return self.data_dir
        
        download_funcs = {
            'gowalla': self.download_gowalla,
            'yelp2018': self.download_yelp2018,
            'amazon-book': self.download_amazon_book,
            'alibaba-ifashion': self.download_alibaba_ifashion,
            'movielens-10m': self.download_movielens_10m,
            'movielens-1m': self.download_movielens_1m,
            'movielens-100k': self.download_movielens_100k,
        }
        
        if dataset not in download_funcs:
            raise ValueError(f"Unknown dataset: {dataset}. "
                           f"Available: {list(download_funcs.keys())}")
        
        return download_funcs[dataset]()
    
    def list_datasets(self):
        """Print available datasets and their status"""
        print("\nAvailable Datasets:")
        print("-" * 70)
        
        for name, info in self.DATASETS.items():
            dataset_dir = os.path.join(self.data_dir, name)
            train_file = os.path.join(dataset_dir, 'train.txt')
            
            status = "✓ Downloaded" if os.path.exists(train_file) else "✗ Not downloaded"
            
            print(f"  {name:<20} {status:<20} {info['description']}")
        
        print("-" * 70)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets for CausalShapGNN')
    parser.add_argument('--dataset', type=str, default='all',
                       help='Dataset to download (gowalla, yelp2018, amazon-book, '
                            'alibaba-ifashion, movielens-10m, movielens-1m, '
                            'movielens-100k, or all)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store datasets')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    
    args = parser.parse_args()
    
    downloader = DataDownloader(args.data_dir)
    
    if args.list:
        downloader.list_datasets()
    else:
        downloader.download(args.dataset)
        downloader.list_datasets()