import json
from zipfile import ZipFile
from data.loaders.utils import download_public_file, QADataset, QARecord, DatasetSplit

def _get_answer_string(answer):
    if isinstance(answer, bool):
        return "Yes" if answer else "No"
    else:
        return answer

def _get_answer_aliases(answer):
    answer = _get_answer_string(answer)
    if answer.lower().startswith("y"):
        aliases = ["Yes", "Correct", "True", "Positive"]
    else:
        aliases = ["No", "Incorrect", "False", "Negative"]
    # add casing and punctuations:
    result = []
    for x in aliases:
        result.append(x)
        result.append(x.lower())
        result.append(x+".")
        result.append(x.lower()+".")
    return result


class StrategyQA(QADataset):
    """
        The dataset instance expert in loading and serving the StrategyQA dataset.
    """
    def __init__(self, config) -> None:
        # self.type = config["Dataset"]["type"]
        self.dataset_url = "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip"
        self.checkpoint_path = config["Experiment"]["checkpoint_path"]
        self.dataset_zip_file = f"strategy_qa.zip"
        download_public_file(self.dataset_url, f"{self.checkpoint_path}/{self.dataset_zip_file}")
        self.split = DatasetSplit.from_str(config["Dataset"]["split"])
        self.data = None

    @property
    def current_file(self):
        if self.split == DatasetSplit.TRAIN:
            return "strategyqa_train.json"
        elif self.split == DatasetSplit.DEV:
            return "strategyqa_train_filtered.json"
        elif self.split == DatasetSplit.TEST:
            return "strategyqa_test.json" # the questions are unanswered!
        else:
            raise ValueError(f"Invalid split {self.split}")

    def _load_data(self):
        with ZipFile(f"{self.checkpoint_path}/{self.dataset_zip_file}", 'r') as zip_ref:
            with zip_ref.open(self.current_file) as f:
                self.data = json.load(f)

    def __iter__(self):
        if self.data is None or len(self.data) == 0:
            self._load_data()
        return self

    def __next__(self):
        try:
            record = self.data.pop()
            return QARecord(
                question=self.normalize_question(record["question"]),
                entity_annotations=None,
                answer=_get_answer_string(record["answer"]) if "answer" in record else None,
                answer_aliases=_get_answer_aliases(record["answer"]) if "answer" in record else None,
                answer_entity_name=None,
                dataset=f"strategy_qa",
                split=self.split.value
            )
        except IndexError:
            raise StopIteration
        except Exception as e:
            print(f"Error: {e}")
            return self.__next__()