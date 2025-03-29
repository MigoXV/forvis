import io
import os
from pathlib import Path

import imageio
from minio import Minio

from forvis.datasets.minio_config import MinIOConfig


class MinIOImageDriver:

    def __init__(self, minio_config: MinIOConfig):
        self.endpoint = minio_config.endpoint
        self.access_key = minio_config.access_key
        self.secret_key = minio_config.secret_key
        self.secure = minio_config.secure
        self.bucket = minio_config.bucket

        self.first_batch = True
        self._client = None
        self.pid = None  # self.pid为创建客户端的进程id，用于检测是否为子进程

    @property
    def client(self):
        """进程安全的MinIO客户端"""
        if self._client is None:
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
            self.pid = os.getpid()
        if os.getpid() != self.pid:
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
            self.pid = os.getpid()
        return self._client

    def get_image(self, image_path: Path):
        """从MinIO中获取数据"""
        image = imageio.imread(
            io.BytesIO(self.client.get_object(self.bucket, str(image_path)).read()),
        )
        if self.first_batch:
            self.clear_cache()
        return image

    def clear_cache(self):
        """清空MinIO客户端缓存"""
        self._client = None
        self.pid = None
