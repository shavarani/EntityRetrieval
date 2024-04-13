import json
from zipfile import ZipFile
from data.loaders.utils import download_public_file, QADataset, QARecord, DatasetSplit

class FactoidQA(QADataset):
    """
        The dataset instance expert in loading and serving the Question-Answer Dataset from https://www.cs.cmu.edu/~ark/QA-data/.
        Dataset description from the website:
            Manually-generated factoid questions from Wikipedia articles, and manually-generated answers to these questions,
            for use in academic research. These data were collected by Noah Smith, Michael Heilman, Rebecca Hwa, Shay Cohen, Kevin Gimpel,
            and many students at Carnegie Mellon University and the University of Pittsburgh between 2008 and 2010.
            Version 1.2 released August 23, 2013.
            Dataset is taken from https://www.cs.cmu.edu/~ark/QA-data/data/Question_Answer_Dataset_v1.2.tar.gz
    """
    def __init__(self, config) -> None:
        self.dataset_url = "https://1sfu-my.sharepoint.com/:u:/g/personal/sshavara_sfu_ca/EaFkI0ecTM1As54ocMo5BFEBLBRsa7k1iWdQc3bpcrgEKA?e=UZSR6F&download=1"
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        self.dataset_zip_file = "factoid_qa.zip"
        download_public_file(self.dataset_url, f"{self.checkpoint_path}/{self.dataset_zip_file}")
        self.data = None

    @property
    def current_file(self):
        return 'data.jsonl'

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
            answers = [x for x in record["answers"]]
            return QARecord(
                question=self.normalize_question(record["question"]),
                entity_annotations=record["titles"],
                answer=answers[0] if answers else None,
                answer_aliases=answers,
                answer_entity_name=None,
                dataset=f"factoid_qa",
                split=DatasetSplit.TRAIN.value
            )
        except IndexError:
            raise StopIteration
        except Exception as e:
            print(f"Error: {e}")
            return self.__next__()