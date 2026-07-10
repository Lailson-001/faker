
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    base_url: str = "https://fakestoreapi.com"
    output_dir: Path = field(default_factory=lambda: Path("data"))
    raw_dir_name: str = "raw"
    processed_dir_name: str = "processed"
    requests_timeout: int = 10
    rate_limit_delay: float = 0.5
    user_agent: str = "python-fakestore-ETL/2.0"

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / self.raw_dir_name

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / self.processed_dir_name

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Persistência — só sabe salvar, não sabe de onde os dados vieram
# ---------------------------------------------------------------------------

class RawDataStorage:

    def __init__(self, config: Config):
        self.config = config
        self.config.ensure_dirs()

    def save_raw_json(self, data: Any, name: str) -> Path:
        filepath = self.config.raw_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "extracted_at": datetime.now().isoformat(),
                    "source": "fakestore_api",
                    "data": data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Raw salvo: {filepath}")
        return filepath

    def save_parquet(
        self,
        df: pd.DataFrame,
        name: str,
        compression: str = "snappy",
    ) -> Path:
        filepath = self.config.processed_dir / f"{name}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath, compression=compression, use_dictionary=True)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Parquet salvo: {filepath} ({size_mb:.2f} MB)")

        self._save_schema(df, name)
        return filepath

    def _save_schema(self, df: pd.DataFrame, name: str) -> None:
        schema_file = self.config.processed_dir / f"{name}_schema.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "shape": df.shape,
                    "created_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Extração — só sabe conversar com a API, não sabe nada sobre arquivos
# ---------------------------------------------------------------------------

class FakeStoreAPI:

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})

    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Método único de requisição — evita repetir headers/timeout/sleep em cada get_*."""
        url = f"{self.config.base_url}/{endpoint}"
        logger.info(f"GET {url}")
        response = self.session.get(
            url,
            params=params,
            timeout=self.config.requests_timeout,
        )
        response.raise_for_status()
        time.sleep(self.config.rate_limit_delay)
        return response.json()

    def get_products(self) -> list[dict]:
        return self._get("products")

    def get_categories(self) -> list[str]:
        return self._get("products/categories")

    def get_products_by_category(self, category: str) -> list[dict]:
        return self._get(f"products/category/{category}")

    def get_users(self) -> list[dict]:
        return self._get("users")

    def get_carts(self) -> list[dict]:
        return self._get("carts")


# ---------------------------------------------------------------------------
# Orquestração — combina API + Storage, não faz o trabalho sujo ela mesma
# ---------------------------------------------------------------------------

class FakeStoreETL:

    def __init__(self, api: FakeStoreAPI, storage: RawDataStorage):
        self.api = api
        self.storage = storage

    def extract_and_save_products(self) -> list[dict]:
        products = self.api.get_products()
        self.storage.save_raw_json(products, "products")
        return products

    def extract_and_save_categories(self) -> list[dict]:
        categories = self.api.get_categories()
        detailed = []
        for cat in categories:
            products = self.api.get_products_by_category(cat)
            detailed.append({"category": cat, "products": products})
        self.storage.save_raw_json(detailed, "categories")
        return detailed

    def extract_and_save_users(self) -> list[dict]:
        users = self.api.get_users()
        self.storage.save_raw_json(users, "users")
        return users

    def extract_and_save_carts(self) -> list[dict]:
        carts = self.api.get_carts()
        self.storage.save_raw_json(carts, "carts")
        return carts

    def run(self) -> dict[str, list[dict]]:
        logger.info("Iniciando extração da FakeStore API")
        start = time.time()

        results = {
            "products": self.extract_and_save_products(),
            "categories": self.extract_and_save_categories(),
            "users": self.extract_and_save_users(),
            "carts": self.extract_and_save_carts(),
        }

        elapsed = time.time() - start
        logger.info(f"Extração concluída em {elapsed:.2f}s")
        return results


# ---------------------------------------------------------------------------
# Uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config(output_dir=Path("data"))
    api = FakeStoreAPI(config=config)
    storage = RawDataStorage(config=config)
    etl = FakeStoreETL(api, storage)

    dados = etl.run()

    print("\n📦 Dados brutos salvos em:", config.raw_dir.absolute())
    for nome, conteudo in dados.items():
        print(f"  ✓ {nome}: {len(conteudo)} registros")