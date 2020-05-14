"""Interactive Labeler

This module includes an InteractiveLabeler.
"""
from six.moves import input

from libact.base.interfaces import Labeler
from libact.utils import inherit_docstring_from


class InteractiveLabeler(Labeler):
    """Interactive Labeler

    InteractiveLabeler is a Labeler object that shows the feature through image
    using matplotlib and lets human label each feature through command line
    interface.

    Parameters
    ----------
    label_name: list
        Let the label space be from 0 to len(label_name)-1, this list
        corresponds to each label's name.
    """

    def __init__(self, **kwargs):
        self.label_name = kwargs.pop('label_name', None)
        self.n_query_per_batch = kwargs.pop('n_query_per_batch', 5)


    def check_labled_results(lbl_res=None):
        pass


    @inherit_docstring_from(Labeler)
    def label(self):
        banner = "Enter the list associated label with the samples, separated with space: "
        if self.label_name is not None:
            banner += str(self.label_name) + ' '

        lbl = input(banner)
        lbl = lbl.split(" ")
        while True:
            is_label_name_valid = (
                self.label_name is not None)                # True means valid

            is_each_labeled_res_valid = True                # True means valid
            for item in lbl:
                tmp = item in self.label_name
                is_each_labeled_res_valid = tmp & is_each_labeled_res_valid

            is_align = len(lbl) == self.n_query_per_batch   # True means valid
            if not(is_label_name_valid and is_each_labeled_res_valid and is_align):
                print('Invalid label, please re-enter the associated label.')
                lbl = input(banner)                
            else:
                break

        # Return the symbolic label
        symbolic_labels = []
        for item in lbl:
            symbolic_labels.append(self.label_name.index(item))
        return symbolic_labels
