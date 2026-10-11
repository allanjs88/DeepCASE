"""Run only the original LANL processing experiment."""
from lanl_experiment import main, original_sequences, save_json
import numpy as np
from deepcase import DeepCASE


def run(args, frame, train, validation, test, vocabulary, folder, timed):
    timings = {}
    (context, events), timings["sequence_seconds"] = timed(
        lambda: original_sequences(frame, vocabulary, args.max_length, args.window_seconds))
    context, events = context.to(args.device), events.to(args.device)
    model = DeepCASE(features=len(vocabulary) + 2, max_length=args.max_length)
    model.to(args.device)
    n = len(train)
    _, timings["training_seconds"] = timed(lambda: model.context_builder.fit(
        context[:n], events[:n], epochs=args.epochs, batch_size=128, verbose=True))
    loss = [float(x) for x in model.context_builder.loss_history]
    if len(loss) != args.epochs:
        raise RuntimeError("Original training stopped before all requested epochs completed.")
    save_json(folder / "loss_history.json", loss)
    print("Original: clustering and assigning benign training scores", flush=True)
    _, timings["interpreter_seconds"] = timed(lambda: model.interpreter.fit(
        context[:n], events[:n], scores=np.zeros(n, dtype=int),
        iterations=100, batch_size=128, verbose=True))
    first = len(train) + len(validation)
    print("Original: predicting test events", flush=True)
    raw, timings["prediction_seconds"] = timed(lambda: model.predict(
        context[first:], events[first:], iterations=100, batch_size=128, verbose=True))
    scored = raw >= 0
    predicted = np.where(scored, (raw > 0).astype(int), -1)
    result = test[["event_id", "timestamp", "label"]].copy()
    result["raw_interpreter_score"] = raw
    result["prediction"] = predicted
    result["scored"] = scored
    model.save(folder / "model.pt")
    extra = {"raw_score_counts": {
        str(k): int(v) for k, v in zip(*np.unique(raw, return_counts=True))}}
    return result, predicted, scored, loss, timings, extra


if __name__ == "__main__":
    main({"original": run})
