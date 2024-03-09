from pydantic import BaseModel
from typing import List, Dict, Any, Union
from pathlib import Path
import json
import os
import os.path as osp    


class Record(BaseModel):
    records: List[Dict[str, Any]] = []
    config: Dict[str, Any]
    save_path: Union[str, Path]

    def add_record(self, metrics: Dict[str, Any]):
        self.records.append(metrics)

    def save(self):
        if "device" in self.config:
            self.config.pop("device")
        if "save_dir" in self.config:
            self.config["save_dir"] = str(self.config["save_dir"])
        print(osp.join(self.save_path, "record.json"))
        with open(osp.join(self.save_path, "record.json"), "w") as f:
            # Since CFG can now be directly converted to dict, we use it here
            json.dump({"records": self.records, "config": self.config}, f)

    @classmethod
    def load(cls, file_path: str):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                return cls(**data, save_path=file_path)
        else:
            raise FileNotFoundError(f"No record found at {file_path}")
            
