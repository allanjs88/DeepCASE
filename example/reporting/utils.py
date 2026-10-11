"""Confusion-report helper copied from deepcase/utils.py; core is unchanged."""
import numpy as np
from sklearn.metrics import confusion_matrix as cm


def confusion_report(
        y_true,
        y_pred,
        labels        = None,
        target_names  = None,
        sample_weight = None,
        normalize     = None,
        skip_x        = set(),
        skip_y        = set(),
    ):
    """Print the confusion matrix as a report.

        Parameters
        ----------
        y_true : array-like of shape=(n_samples,)
            Actual labels of evaluated values.

        y_pred : array-like of shape=(n_samples,)
            Predicted labels of evaluated values.

        labels : array-like of shape=(n_labels,), optional
            All different labels to include in the confusion report.
            If none are given, labels are inferred from the y_true and y_pred
            inputs. Labels should at least contain all labels in y_true and
            y_pred, but additional labels may be given.

        target_names : array-like of shape=(n_labels,), optional
            If given, use names pressented by target_names for each
            corresponding label.

        sample_weight : array-like of shape=(n_samples,), optional
            If given, weigh input by given sample_weight.

        normalize : {‘true’, ‘pred’, ‘all’}, default=None
            Normalizes confusion matrix over the true (rows), predicted
            (columns) conditions or all the population. If None, confusion
            matrix will not be normalized.

        skip_x : set(), optional
            Set of target_names to skip while printing the columns.

        skip_y : set(), optional
            Set of target_names to skip while printing the rows.

        Returns
        -------
        result : string
            Report detailing the confusion matrix for a given prediction.
        """
    # Compute matrix
    matrix = cm(
        y_true        = y_true,
        y_pred        = y_pred,
        labels        = labels,
        sample_weight = sample_weight,
        normalize     = normalize,
    )

    if target_names is not None:
        assert labels       is not None
        assert target_names is not None
        assert len(labels) == len(target_names)

        # Add labels to matrix
        matrix = np.concatenate(([target_names], matrix))
        matrix = np.concatenate(([["T\\P"] + target_names], matrix.T)).T

    # Compute width of rows
    width = np.vectorize(len)(matrix).max()

    # Transform to string
    result = ""
    mask_x = [i for i, x in enumerate(matrix[0   ]) if x not in skip_x]
    mask_y = [i for i, x in enumerate(matrix[:, 0]) if x not in skip_y]
    for row in matrix[mask_y]:
        result += "\t".join(
            "{:>{width}}".format(element, width=width)
            for element in row[mask_x]
        ) + '\n'

    return result



