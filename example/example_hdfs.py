# Other imports
from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .plotting import ResultsPlotter
else:
    from plotting import ResultsPlotter

# DeepCASE Imports
from deepcase.preprocessing   import Preprocessor
from deepcase.context_builder import ContextBuilder


TRAINING_EPOCHS = 100


if __name__ == "__main__":
    parser = ArgumentParser(description="Train and evaluate HDFS next-event prediction.")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit input sequences for a quick run (default: all).")
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS,
                        help="Training epochs (default: 100).")
    parser.add_argument("--no-graphs", action="store_true",
                        help="Skip generating, saving, and displaying plots.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots without opening interactive windows.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for plots (default: the example directory).")
    args = parser.parse_args()
    if args.epochs < 1 or (args.nrows is not None and args.nrows < 1):
        parser.error("--epochs and --nrows must be positive integers")
    if args.no_show:
        plt.switch_backend("Agg")

    example_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir is not None else example_dir
    if not args.no_graphs:
        output_dir.mkdir(parents=True, exist_ok=True)
    hdfs_path = example_dir / "data" / "hdfs" / "hdfs_test_normal"

    ########################################################################
    #                             Loading data                             #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        length  = 10,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
    )

    # Load data from file
    context, events, labels, mapping = preprocessor.text(
        path    = hdfs_path,
        nrows   = args.nrows,
        verbose = True,
    )

    # In case no labels are provided, set labels to -1
    # IMPORTANT: If no labels are provided, make sure to manually set the labels
    # before calling the interpreter.score_clusters method. Otherwise, this will
    # raise an exception, because scores == NO_SCORE cannot be computed.
    if labels is None:
        labels = np.full(events.shape[0], -1, dtype=int)

    # Cast to cuda if available
    if torch.cuda.is_available():
        print("Cuda is available")
        events  = events .to('cuda')
        context = context.to('cuda')
    else:
        print ("Cuda is not available")

    ########################################################################
    #                            Splitting data                            #
    ########################################################################

    # Split into train and test sets (20:80) by time - assuming events are ordered chronologically
    events_train  = events [:events.shape[0]//5 ]
    events_test   = events [ events.shape[0]//5:]

    context_train = context[:events.shape[0]//5 ]
    context_test  = context[ events.shape[0]//5:]

    labels_train  = labels [:events.shape[0]//5 ]
    labels_test   = labels [ events.shape[0]//5:]

    ########################################################################
    #                       Training ContextBuilder                        #
    ########################################################################

    # Create ContextBuilder
    context_builder = ContextBuilder(
        input_size    =  30,   # Number of input features to expect
        output_size   =  30,   # Same as input size
        hidden_size   = 128,   # Number of nodes in hidden layer, in paper we set this to 128
        max_length    = 10,    # Length of the context, should be same as context in Preprocessor
    )

    # Cast to cuda if available
    if torch.cuda.is_available():
        context_builder = context_builder.to('cuda')

    # Train the ContextBuilder
    context_builder.fit(
        X             = context_train,               # Context to train with
        y             = events_train.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
        epochs        = args.epochs,                # Number of epochs to train with
        batch_size    = 128,                         # Number of samples in each training batch, in paper this was 128
        learning_rate = 0.01,                        # Learning rate to train with, in paper this was 0.01
        verbose       = True,                        # If True, prints progress
    )

    ########################################################################
    #                  Get prediction from ContextBuilder                  #
    ########################################################################

    # Use context builder to predict confidence
    confidence, _ = context_builder.predict(
        X = context_test
    )

    # Get confidence of the next step, seq_len 0 (n_samples, seq_len, output_size)
    confidence = confidence[:, 0]
    # Get confidence from log confidence
    confidence = confidence.exp()
    # Get prediction as maximum confidence
    y_pred = confidence.argmax(dim=1)

    ########################################################################
    #                          Perform evaluation                          #
    ########################################################################

    # Get test and prediction as numpy array
    y_test = events_test.cpu().numpy()
    y_pred = y_pred     .cpu().numpy()

    # Print classification report
    print(classification_report(
        y_true = y_test,
        y_pred = y_pred,
        digits = 4,
    ))

    ########################################################################
    #                       Matplotlib visualization                       #
    ########################################################################

    ResultsPlotter.plot_prediction_results(
        y_test      = y_test,
        y_pred      = y_pred,
        output_path = output_dir / "hdfs_prediction_results.png",
        title       = "HDFS Event Prediction Confusion Matrix",
        show        = not args.no_show,
        generate_graph = not args.no_graphs,
    )

    ResultsPlotter.plot_training_loss_history(
        loss_history = context_builder.loss_history,
        output_path = output_dir / "hdfs_training_loss_summary.png",
        show        = not args.no_show,
        generate_graph = not args.no_graphs,
    )
