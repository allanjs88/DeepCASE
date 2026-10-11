"""Run only the gnn LANL processing experiment."""
from lanl_experiment import main, save_json
import numpy as np
import pandas as pd
from deepcase_gnn import DeepCASEGNN


def run(args, frame, train, validation, test, vocabulary, folder, timed):
    timings = {}
    model = DeepCASEGNN(max_length=args.max_length, window_seconds=args.window_seconds,
                        gnn_layers=2, seed=args.seed, device=args.device)
    print("GNN: graph construction runs on CPU; training prints after each epoch", flush=True)
    _, timings["training_seconds"] = timed(lambda: model.fit(
        train, epochs=args.epochs, batch_size=4, verbose=True))
    loss = [float(x) for x in model.loss_history]
    save_json(folder / "loss_history.json", loss)
    print("GNN: calibrating validation threshold", flush=True)
    _, timings["calibration_seconds"] = timed(lambda: model.calibrate(
        validation, history=train, quantile=0.99, batch_size=4))
    print("GNN: predicting test events", flush=True)
    result, timings["prediction_seconds"] = timed(lambda: model.predict(
        test, history=pd.concat([train, validation]), batch_size=4))
    scored = np.ones(len(result), dtype=bool)
    predicted = result.prediction.to_numpy()
    result["scored"] = scored
    model.save(folder / "model.pt")
    return result, predicted, scored, loss, timings, {"threshold": float(model.interpreter.threshold)}


if __name__ == "__main__":
    main({"gnn": run})
