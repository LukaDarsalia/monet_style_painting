"""
S3 utilities for uploading and downloading files/folders with tar.gz compression support.
"""

import datetime
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Union

import boto3
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv('.env')


def get_s3_loader(bucket_name: str) -> 'S3DataLoader':
    """Create S3DataLoader with credentials from environment variables."""
    return S3DataLoader(
        bucket=bucket_name,
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )


def generate_folder_name() -> str:
    """Generate timestamp-based folder name (YYYY-MM-DD_HH-MM-SS)."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class S3DataLoader:
    """Utility class for S3 upload/download operations with tar.gz compression."""

    def __init__(self, bucket: str, access_key: str, secret_key: str):
        self.bucket = bucket
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def upload_file(self, filepath: Union[str, Path], s3_key: Optional[str] = None) -> str:
        """Upload a single file to S3."""
        filepath = Path(filepath)
        s3_key = s3_key or str(filepath)
        self.s3_client.upload_file(str(filepath), self.bucket, s3_key)
        return s3_key

    def download_file(self, s3_key: str, local_path: Union[str, Path]) -> Path:
        """Download a single file from S3."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, s3_key, str(local_path))
        return local_path

    def upload_as_tarball(self, folder_path: Union[str, Path], s3_key: Optional[str] = None) -> str:
        """
        Compress folder to tar.gz and upload to S3.
        
        Args:
            folder_path: Local folder to compress and upload
            s3_key: S3 key for tarball. Defaults to {folder_path}.tar.gz
            
        Returns:
            S3 key where tarball was uploaded
        """
        folder_path = Path(folder_path)
        s3_key = s3_key or f"{folder_path}.tar.gz"

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            print(f"  Compressing {folder_path}...")
            with tarfile.open(tmp_path, 'w:gz') as tar:
                all_files = list(folder_path.rglob("*"))
                for file_path in tqdm(all_files, desc="  Compressing"):
                    if not file_path.is_dir():
                        arcname = file_path.relative_to(folder_path.parent)
                        tar.add(file_path, arcname=arcname)

            file_size = os.path.getsize(tmp_path)
            print(f"  Archive size: {file_size / (1024*1024):.2f} MB")
            print(f"  Uploading to s3://{self.bucket}/{s3_key}...")
            
            self.s3_client.upload_file(tmp_path, self.bucket, s3_key)
            print(f"  Upload complete")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return s3_key

    def download_and_extract_tarball(self, s3_key: str, extract_to: Union[str, Path] = ".") -> Path:
        """
        Download tar.gz from S3 and extract it.
        
        Args:
            s3_key: S3 key of the tarball
            extract_to: Directory to extract to
            
        Returns:
            Path where files were extracted
        """
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            print(f"  Downloading s3://{self.bucket}/{s3_key}...")
            self.s3_client.download_file(self.bucket, s3_key, tmp_path)
            
            file_size = os.path.getsize(tmp_path)
            print(f"  Downloaded: {file_size / (1024*1024):.2f} MB")

            print(f"  Extracting to {extract_to}...")
            with tarfile.open(tmp_path, 'r:gz') as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc="  Extracting"):
                    tar.extract(member, extract_to)
            print(f"  Extracted {len(members)} files")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return extract_to
