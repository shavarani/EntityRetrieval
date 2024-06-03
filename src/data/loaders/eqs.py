import json
from zipfile import ZipFile
from data.loaders.utils import download_public_file, QADataset, QARecord, DatasetSplit


class EntityQuestions(QADataset):
    """
        The dataset instance expert in loading and serving the EntityQuestions dataset with data files from:
         https://aclanthology.org/2021.emnlp-main.496.pdf
    """
    def __init__(self, config, load_selected_questions=True) -> None:
        if load_selected_questions:
            _url = "ER1H-msJSo9AoTMJ5u0xeaIBgLiMAq-4GwMglGLNjigMWQ?e=on3rYg"
            self.dataset_zip_file = f"selected_entity_questions.zip"
        else:
            _url = "EdPEdLlFu7hEltetjFLbIFkBw936g-3ty-1UZtJ_Ej1TsA?e=JNHw9q"
            self.dataset_zip_file = f"entity_questions.zip"
        self.dataset_url = f"https://1sfu-my.sharepoint.com/:u:/g/personal/sshavara_sfu_ca/{_url}&download=1"
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        download_public_file(self.dataset_url, f"{self.checkpoint_path}/{self.dataset_zip_file}")
        self.split = DatasetSplit.from_str(config["Dataset"]["split"])
        self.data = None

    @property
    def current_file(self):
        if self.split == DatasetSplit.TRAIN:
            return "train.jsonl"
        elif self.split == DatasetSplit.DEV:
            return "dev.jsonl"
        elif self.split == DatasetSplit.TEST:
            return "test.jsonl"
        else:
            raise ValueError(f"Invalid split {self.split}")

    def _load_data(self):
        zip_ref = ZipFile(f"{self.checkpoint_path}/{self.dataset_zip_file}", 'r')
        self.data = zip_ref.open(self.current_file)

    def __iter__(self):
        if self.data is None or len(self.data) == 0:
            self._load_data()
        return self

    def __next__(self):
        try:
            next_line = self.data.readline()
            if not next_line:
                self.data.close()
                self.data = None
                raise IndexError
            record = json.loads(next_line)
            answers = record["answers"]
            return QARecord(
                question=self.normalize_question(record["question"]),
                entity_annotations=[record["entity"]],
                answer=answers[0] if answers else None,
                answer_aliases=answers,
                answer_entity_name=None,
                dataset="entity_questions",
                split=self.split.value
            )
        except IndexError:
            raise StopIteration
        except Exception as e:
            print(f"Error: {e}")
            return self.__next__()