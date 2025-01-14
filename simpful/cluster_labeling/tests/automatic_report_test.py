from simpful import FuzzySystem

from simpful.cluster_labeling.cluster_labeling import approximate_fs_labels
from simpful.cluster_labeling.tests.Cement_Simpful_code import get_cement_fs


def main():
    # # ---------------------------------------
    fs: FuzzySystem = get_cement_fs()
    approximate_fs_labels(fs, output_path="output", generate_report=False)
    # ---------------------------------------


if __name__ == '__main__':
    main()
