import argparse
import sys
import time
from pathlib import Path

import pipeline as pl

try:
    sys.stdout.reconfigure(encoding="utf-8")  # чтобы русский не ломался в консоли Windows
except Exception:
    pass


def evaluate_on_calibration(retriever, calibration, methods):
    queries = calibration["query_text"].tolist()
    relevants = [set(pl.parse_ground_truth(g)) for g in calibration["ground_truth"]]
    print("\nMAP@10 на калибровке:")
    for method in methods:
        t0 = time.time()
        preds = retriever.rank(queries, method=method, top_n=10)
        score = pl.mean_average_precision_at_k(preds, relevants, k=10)
        print(f"  {method:<8} {score:.4f}   ({time.time() - t0:.1f} c)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Files/candidate_data")
    parser.add_argument("--out", default="answer.csv")
    parser.add_argument("--method", default="hybrid", choices=["bm25", "dense", "hybrid", "rerank"])
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    pl.set_seed()
    cfg = pl.Config(verbose=not args.quiet)
    final_method = "bm25" if args.no_dense else args.method
    use_dense = final_method in ("dense", "hybrid", "rerank")

    print(f"Загрузка данных из {args.data_dir} ...")
    articles, calibration, test = pl.load_data(args.data_dir)
    print(f"  статей: {len(articles)}, тестовых запросов: {len(test)}")

    t0 = time.time()
    retriever = pl.Retriever(articles, cfg).build(use_dense=use_dense)
    print(f"  индекс готов за {time.time() - t0:.1f} c")

    if not args.no_eval and calibration is not None:
        methods = ["bm25"]
        if use_dense:
            methods += ["dense", "hybrid"]
        if final_method == "rerank":
            methods.append("rerank")
        if final_method not in methods:
            methods.append(final_method)
        evaluate_on_calibration(retriever, calibration, methods)

    print(f"\nРанжирование test.f методом '{final_method}' ...")
    preds = retriever.rank(test["query_text"].tolist(), method=final_method, top_n=cfg.top_n)
    answer = pl.format_answers(test["query_id"].tolist(), preds)

    errors = pl.validate_answers(answer, test, articles, top_n=cfg.top_n)
    if errors:
        print("\nОшибки валидации answer.csv:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)

    answer.to_csv(Path(args.out), index=False)
    print(f"\nГотово: {args.out} ({len(answer)} строк)")


if __name__ == "__main__":
    main()
