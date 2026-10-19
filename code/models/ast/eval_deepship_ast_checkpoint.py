import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from train_eval_deepship_ast import (  # noqa: E402
    PROJECT_ROOT,
    aggregate_recordings,
    build_model,
    build_prototypes,
    class_names_from_index,
    evaluate_closed_set,
    extract_embeddings,
    load_ship_to_big,
    make_loaders,
    pairwise_scores,
    plot_scores,
    retrieval,
    set_seed,
    verification_metrics,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--init-type", choices=["pretrained_audioset", "random_init"], required=True)
    ap.add_argument("--pretrained-model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--feature-index", required=True)
    ap.add_argument(
        "--split-csv",
        default=str(
            PROJECT_ROOT
            / "split_protocols"
            / "deepship"
            / "ship_identity"
            / "min5_median_split_20260511_163606"
            / "split_detail.csv"
        ),
    )
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_index = Path(args.feature_index)
    class_names = class_names_from_index(feature_index)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    _, val_dl, test_dl = make_loaders(feature_index, args.batch_size, args.num_workers, class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.init_type, args.pretrained_model_name, class_names).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_m, val_pc = evaluate_closed_set(model, val_dl, device, class_names, out_dir / "val_predictions.csv")
    test_m, test_pc = evaluate_closed_set(model, test_dl, device, class_names, out_dir / "test_predictions.csv")
    val_pc.to_csv(out_dir / "val_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    test_pc.to_csv(out_dir / "test_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    val_meta, val_emb = extract_embeddings(model, val_dl, device)
    test_meta, test_emb = extract_embeddings(model, test_dl, device)
    np.save(out_dir / "val_embeddings.npy", val_emb)
    np.save(out_dir / "test_embeddings.npy", test_emb)
    val_meta.to_csv(out_dir / "val_embedding_meta.csv", index=False, encoding="utf-8-sig")
    test_meta.to_csv(out_dir / "test_embedding_meta.csv", index=False, encoding="utf-8-sig")

    ship_to_big = load_ship_to_big(Path(args.split_csv))
    proto_meta, proto_emb = build_prototypes(val_meta, val_emb, ship_to_big)
    np.save(out_dir / "val_prototypes.npy", proto_emb)
    proto_meta.to_csv(out_dir / "val_prototype_meta.csv", index=False, encoding="utf-8-sig")

    score_df = pairwise_scores(test_meta, test_emb, proto_meta, proto_emb)
    score_df.to_csv(out_dir / "verification_scores.csv", index=False, encoding="utf-8-sig")
    verif_df = verification_metrics(score_df)
    verif_df.to_csv(out_dir / "verification_metrics_by_pair_type.csv", index=False, encoding="utf-8-sig")
    plot_scores(score_df, out_dir / "score_distribution.png")

    seg_rank, seg_m = retrieval(test_meta, test_emb, proto_meta, proto_emb, "segment")
    rec_meta, rec_emb = aggregate_recordings(test_meta, test_emb)
    rec_rank, rec_m = retrieval(rec_meta, rec_emb, proto_meta, proto_emb, "recording")
    seg_rank.to_csv(out_dir / "retrieval_segment_rankings.csv", index=False, encoding="utf-8-sig")
    rec_rank.to_csv(out_dir / "retrieval_recording_rankings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([seg_m]).to_csv(out_dir / "retrieval_segment_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([rec_m]).to_csv(out_dir / "retrieval_recording_metrics.csv", index=False, encoding="utf-8-sig")

    verif = verif_df.set_index("pair_set").to_dict(orient="index")
    summary = {
        "model_name": "AST-AudioSet" if args.init_type == "pretrained_audioset" else "AST-random-init",
        "init_type": args.init_type,
        "checkpoint": str(checkpoint.resolve()),
        "closed_set_val_accuracy": val_m["accuracy"],
        "closed_set_val_macro_F1": val_m["macro_f1"],
        "closed_set_test_accuracy": test_m["accuracy"],
        "closed_set_test_macro_F1": test_m["macro_f1"],
        "zero_recall_ship_count": test_m["zero_recall_ship_count"],
        "zero_recall_ship_names": ";".join(test_pc.loc[test_pc["recall"] == 0, "class_name"].tolist()),
        "AUC_all": verif["all"]["AUC"],
        "AUC_hard": verif["hard"]["AUC"],
        "AUC_easy": verif["easy"]["AUC"],
        "EER_all": verif["all"]["EER"],
        "EER_hard": verif["hard"]["EER"],
        "EER_easy": verif["easy"]["EER"],
        "positive_mean_similarity": verif["all"]["positive_mean_similarity"],
        "hard_negative_mean_similarity": verif["all"]["hard_negative_mean_similarity"],
        "easy_negative_mean_similarity": verif["all"]["easy_negative_mean_similarity"],
        "hard_negative_gap": verif["all"]["hard_negative_gap"],
        "easy_negative_gap": verif["all"]["easy_negative_gap"],
        "Segment_Recall@1": seg_m["Recall@1"],
        "Segment_Recall@5": seg_m["Recall@5"],
        "Segment_mAP": seg_m["mAP"],
        "Recording_Top1": rec_m["Recall@1"],
        "Recording_Top5": rec_m["Recall@5"],
        "Recording_mAP_or_MRR": rec_m["MRR"],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        f"# {summary['model_name']} DeepShip Evaluation",
        "",
        "- Split: DeepShip 25-class min5 median recording-wise split.",
        "- Downstream matching: validation prototypes, test queries.",
        "",
        f"- Test accuracy: {summary['closed_set_test_accuracy']:.6f}",
        f"- Test macro-F1: {summary['closed_set_test_macro_F1']:.6f}",
        f"- Zero-recall ship count: {summary['zero_recall_ship_count']}",
        f"- AUC-all: {summary['AUC_all']:.6f}",
        f"- AUC-hard: {summary['AUC_hard']:.6f}",
        f"- Segment Recall@1: {summary['Segment_Recall@1']:.6f}",
        f"- Segment Recall@5: {summary['Segment_Recall@5']:.6f}",
        f"- Segment mAP/MRR: {summary['Segment_mAP']:.6f}",
        f"- Recording Top-1: {summary['Recording_Top1']:.6f}",
        f"- Recording Top-5: {summary['Recording_Top5']:.6f}",
        f"- Recording mAP/MRR: {summary['Recording_mAP_or_MRR']:.6f}",
    ]
    (out_dir / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
