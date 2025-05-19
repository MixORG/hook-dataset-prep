import json
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

def group_into_sentences(words):
    sentences, current = [], []
    for w in words:
        token = w.get("punctuated_word") or w.get("word")
        if not token:
            continue
        current.append({
            "start": w.get("start"),
            "end": w.get("end"),
            "text": token
        })
        if token.endswith(('.', '?', '!')):
            sentence_text = " ".join(x["text"] for x in current)
            sentences.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": sentence_text,
                "words": current.copy()
            })
            current = []
    if current:
        sentence_text = " ".join(x["text"] for x in current)
        sentences.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": sentence_text,
            "words": current.copy()
        })
    return sentences

def exact_match(hook_sents, main_sents, vid):
    rows = []
    for i, sent in enumerate(hook_sents):
        words = [w["text"] for w in sent["words"]]
        n = len(words)
        for j in range(len(main_sents) - n + 1):
            window = main_sents[j:j+n]
            if all(window[k]["text"] == words[k] for k in range(n)):
                rows.append({
                    "youtube_id": vid,
                    "start_timestamp": f"{window[0]['start']:.3f}",
                    "end_timestamp": f"{window[-1]['end']:.3f}",
                    "matched_text": sent["text"],
                    "hook_original_order": i + 1
                })
                break
    return pd.DataFrame(rows)

def fuzzy_match(hook_sents, main_sents, vid, threshold=0.90):
    rows = []
    for i, sent in enumerate(hook_sents):
        sent_text = sent["text"]
        n = len(sent["words"])
        best_score, best_match = 0, None
        for j in range(len(main_sents) - n + 1):
            window = main_sents[j:j + n]
            window_text = " ".join(w.get("text", "") for w in window)
            score = SequenceMatcher(None, sent_text.lower(), window_text.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = j
        if best_score >= threshold and best_match is not None:
            rows.append({
                "youtube_id": vid,
                "hook_original_order": i + 1,
                "matched_text": sent_text,
                "match_score": round(best_score * 100, 2),
                "start_timestamp": f"{main_sents[best_match]['start']:.3f}",
                "end_timestamp": f"{main_sents[best_match + n - 1]['end']:.3f}"
            })
    return pd.DataFrame(rows)

def create_final_transcript(main_sents, matches_df, hook_sents, vid):
    main_df = pd.DataFrame([
        {
            "transcript_id": vid,
            "start_timestamp": f"{s['start']:.3f}",
            "end_timestamp": f"{s['end']:.3f}",
            "text": s["text"],
            "is_hook": 0,
            "hook_rank": -1
        }
        for s in main_sents
    ])
    if not matches_df.empty and "hook_original_order" in matches_df.columns:
        matches_df["start_float"] = matches_df["start_timestamp"].astype(float)
        matches_df["end_float"] = matches_df["end_timestamp"].astype(float)
        for i, row in main_df.iterrows():
            for _, match in matches_df.iterrows():
                if float(row["start_timestamp"]) <= match["end_float"] and float(row["end_timestamp"]) >= match["start_float"]:
                    main_df.at[i, "is_hook"] = 1
                    main_df.at[i, "hook_rank"] = match["hook_original_order"]
                    break
    main_df["sentence_id"] = range(1, len(main_df) + 1)
    return main_df[["transcript_id", "sentence_id", "start_timestamp", "end_timestamp", "text", "is_hook", "hook_rank"]]

def evaluate(hook_sents, matches_df):
    total_hooks = len(hook_sents)
    matched_hooks = matches_df["hook_original_order"].nunique() if not matches_df.empty and "hook_original_order" in matches_df.columns else 0
    precision = matched_hooks / len(matches_df) if len(matches_df) > 0 else 0.0
    recall = matched_hooks / total_hooks if total_hooks > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "total_hooks": total_hooks,
        "matched_hooks": matched_hooks,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2)
    }

# === Main Execution ===
output_dir = Path("outputs")
result_dir = Path("fuzzy_results")
result_dir.mkdir(exist_ok=True)

for folder in output_dir.iterdir():
    if not folder.is_dir():
        continue

    vid = folder.name
    print(f"🔍 Processing {vid}...")

    hook_path = folder / "hook_transcript.json"
    rest_path = folder / "rest_of_transcript.json"

    if not hook_path.exists() or not rest_path.exists():
        print(f"⚠️ Skipping {vid}: Missing JSONs")
        continue

    try:
        with open(hook_path) as f:
            hook_json = json.load(f)
        with open(rest_path) as f:
            rest_json = json.load(f)

        hook_words = hook_json["results"]["channels"][0]["alternatives"][0].get("words", [])
        rest_words = rest_json["results"]["channels"][0]["alternatives"][0].get("words", [])

        if not hook_words or not rest_words:
            print(f"❌ Skipping {vid}: Empty transcripts")
            continue

        hook_sents = group_into_sentences(hook_words)
        main_sents = group_into_sentences(rest_words)

        exact_df = exact_match(hook_sents, main_sents, vid)
        fuzzy_df = fuzzy_match(hook_sents, main_sents, vid, threshold=0.90)

        matches_df = pd.concat([exact_df, fuzzy_df]).drop_duplicates(subset=["start_timestamp", "end_timestamp"])
        if not matches_df.empty and "hook_original_order" in matches_df.columns:
            matches_df.sort_values("hook_original_order", inplace=True)

        eval_metrics = evaluate(hook_sents, matches_df)
        with open(result_dir / f"{vid}_eval.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)

        matched = eval_metrics["matched_hooks"]
        total = eval_metrics["total_hooks"]

        if not matches_df.empty:
            final_df = create_final_transcript(main_sents, matches_df, hook_sents, vid)
            matches_df.to_csv(result_dir / f"{vid}_fuzzy_matches.csv", index=False)
            final_df.to_csv(result_dir / f"{vid}_final.csv", index=False)
            print(f"✅ {vid}: Matched {matched}/{total} hooks — F1: {eval_metrics['f1_score']}")
        else:
            print(f"⚠️ {vid}: No matches found ({matched}/{total}) — only evaluation saved.")

    except Exception as e:
        print(f"❌ Error in {vid}: {e}")
