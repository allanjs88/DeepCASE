# Other imports
from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch

# DeepCASE Imports
from deepcase.preprocessing   import Preprocessor
from deepcase.context_builder import ContextBuilder


TRAINING_EPOCHS = 100


def plot_hdfs_results(y_test, y_pred, output_path, show=True):
    """Plot prediction quality after the HDFS example finishes."""
    classes = np.union1d(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=classes)

    fig, (ax_matrix, ax_counts) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )
    display.plot(
        ax=ax_matrix,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax_matrix.set_title("HDFS Event Prediction Confusion Matrix")
    ax_matrix.tick_params(axis="x", labelrotation=90)

    x = np.arange(classes.shape[0])
    width = 0.4
    actual_counts = np.array([(y_test == event).sum() for event in classes])
    predicted_counts = np.array([(y_pred == event).sum() for event in classes])

    ax_counts.bar(x - width / 2, actual_counts, width, label="Actual")
    ax_counts.bar(x + width / 2, predicted_counts, width, label="Predicted")
    ax_counts.set_title("Actual vs Predicted Event Counts")
    ax_counts.set_xlabel("Event")
    ax_counts.set_ylabel("Number of samples")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(classes, rotation=90)
    ax_counts.legend()

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print("Saved Matplotlib visualization to {}".format(output_path))
    if show:
        plt.show()
    plt.close(fig)


def plot_training_loss_history(loss_history, output_path, show=True):
    """Plot the ContextBuilder training loss from every completed epoch."""
    if len(loss_history) == 0:
        print("No training loss values were collected; skipping loss plot.")
        return

    loss_points = np.asarray(loss_history, dtype=float)
    epoch_points = np.arange(1, loss_points.shape[0] + 1)
    first_loss = loss_points[0]
    final_loss = loss_points[-1]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epoch_points, loss_points, marker="o", linewidth=2.5, color="tab:green")
    ax.fill_between(epoch_points, loss_points, loss_points.min(), alpha=0.15, color="tab:green")

    ax.annotate(
        "First epoch\nloss = {:.4f}".format(first_loss),
        xy=(1, first_loss),
        xytext=(max(1, epoch_points[-1] * 0.08), first_loss + 0.001),
        arrowprops={"arrowstyle": "->", "color": "0.35"},
    )
    ax.annotate(
        "Final epoch\nloss = {:.4f}".format(final_loss),
        xy=(epoch_points[-1], final_loss),
        xytext=(max(1, epoch_points[-1] * 0.62), final_loss + 0.001),
        arrowprops={"arrowstyle": "->", "color": "0.35"},
    )

    ax.set_title("ContextBuilder Training Loss Decrease")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_xlim(1, epoch_points[-1])
    ax.grid(True, linestyle="--", alpha=0.35)

    improvement = first_loss - final_loss
    improvement_percent = improvement / first_loss * 100
    fig.suptitle(
        "Completed {} epochs: loss decreased from {:.4f} to {:.4f} ({:.1f}% lower)"
        .format(epoch_points[-1], first_loss, final_loss, improvement_percent),
        y=1.03,
        fontsize=10,
    )

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print("Saved training loss visualization to {}".format(output_path))
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train and evaluate HDFS next-event prediction.")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit input sequences for a quick run (default: all).")
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS,
                        help="Training epochs (default: 100).")
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

    plot_hdfs_results(
        y_test      = y_test,
        y_pred      = y_pred,
        output_path = output_dir / "hdfs_prediction_results.png",
        show        = not args.no_show,
    )

    plot_training_loss_history(
        loss_history = context_builder.loss_history,
        output_path = output_dir / "hdfs_training_loss_summary.png",
        show        = not args.no_show,
    )
