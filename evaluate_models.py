from config import EVAL_DATA_DIR, EVAL_FILES_DIR
from utils.evaluator import Evaluator


def main():
    evaluator = Evaluator(EVAL_DATA_DIR, EVAL_FILES_DIR)
    evaluator.create_report()
    evaluator.print_report()


if __name__ == '__main__':
    main()
