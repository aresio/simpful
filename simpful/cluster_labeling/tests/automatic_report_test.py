from simpful import FuzzySystem

from simpful.cluster_labeling.cluster_labeling import approximate_fs_labels
from simpful.cluster_labeling.tests.Cement_Simpful_code import get_cement_fs


# from simpful.cluster_labeling.tests.Simpful_code_2rules import wrap_2rules


def main():
    # # ---------------------------------------
    # fs_2_system: FuzzySystem = wrap_2rules()
    fs: FuzzySystem = get_cement_fs()
    approximate_fs_labels(fs)
    # ---------------------------------------


if __name__ == '__main__':
    main()
