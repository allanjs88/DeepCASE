"""Reusable plots for event prediction examples."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class ResultsPlotter:
    """Plot prediction quality and training history for any event dataset."""

    @staticmethod
    def plot_prediction_results(
        y_test, y_pred, output_path, show=True, generate_graph=True,
        title="Event Prediction Confusion Matrix",
    ):
        """Save prediction plots; show controls display, generate_graph skips all plotting."""
        if not generate_graph:
            return

        y_test = np.asarray(y_test)
        y_pred = np.asarray(y_pred)
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
        ax_matrix.set_title(title)
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


    @staticmethod
    def plot_training_loss_history(
        loss_history, output_path, show=True, generate_graph=True,
    ):
        """Save epoch losses; show controls display, generate_graph skips all plotting."""
        if not generate_graph:
            return
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

        ax.set_title("Training Loss Decrease")
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
