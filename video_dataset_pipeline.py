import os
import subprocess
from pathlib import Path
import pandas as pd
from urllib.parse import urlparse, parse_qs
from rapidfuzz import fuzz
import requests
from deepgram import DeepgramClient

DEEPGRAM_API_KEY = ''
DG = DeepgramClient(api_key=DEEPGRAM_API_KEY)


def transcribe_audio(audio_path: Path) -> list:
    """
    Transcribe a local WAV file using Deepgram REST API with increased timeouts.
    Returns a list of segments with start, end (seconds with 3 decimals), and text.
    """
    url = 'https://api.deepgram.com/v1/listen'
    params = {'model': 'nova-3', 'smart_format': 'true'}
    headers = {
        'Authorization': f'Token {DEEPGRAM_API_KEY}',
        'Content-Type': 'audio/wav'
    }
    with open(audio_path, 'rb') as f:
        response = requests.post(
            url, headers=headers, params=params, data=f, timeout=(10, 300)
        )
    response.raise_for_status()
    resp = response.json()

    words = resp.get('results', {}).get('channels', [{}])[0] \
                .get('alternatives', [{}])[0] \
                .get('words', [])

    segments = []
    for w in words:
        text = w.get('punctuated_word', w.get('word', ''))
        segments.append({
            'start': float(f"{w['start']:.3f}"),
            'end':   float(f"{w['end']:.3f}"),
            'text':  text
        })
    return segments


def clean_url(youtube_url: str) -> str:
    """Strip playlist/index params, returns clean watch URL."""
    parsed = urlparse(youtube_url)
    vid = parse_qs(parsed.query).get('v', [None])[0]
    return f'https://www.youtube.com/watch?v={vid}' if vid else youtube_url


def download_audio(youtube_url: str, out_dir: Path,
                   cookies_from_browser: str = None,
                   cookies_file: str = None) -> Path:
    """Download YouTube audio as WAV, with optional cookies."""
    url_clean = clean_url(youtube_url)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    parsed = urlparse(url_clean)
    video_id = parse_qs(parsed.query).get('v', [None])[0]
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url_clean}")
    
    template = out_dir / f'{video_id}.%(ext)s'
    
    cmd = ['yt-dlp', '--no-playlist', '-x', '--audio-format', 'wav', '-o', str(template)]
    if cookies_from_browser:
        cmd += ['--cookies-from-browser', cookies_from_browser]
    if cookies_file:
        cmd += ['--cookies', cookies_file]
    cmd.append(url_clean)
    subprocess.run(cmd, check=True)
    
    ws = list(out_dir.glob(f'{video_id}.wav'))
    if not ws:
        raise FileNotFoundError(f'No WAV found for {youtube_url} with ID {video_id}')
    return ws[0]


def hhmmss_to_seconds(ts: str) -> float:
    p = list(map(int, ts.split(':')))
    while len(p) < 3: p.insert(0, 0)
    return p[0]*3600 + p[1]*60 + p[2]


def seconds_to_hhmmss(sec: float) -> str:
    h = int(sec//3600); m = int((sec%3600)//60); s = int(sec%60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def cut_segment(in_file: Path, start: str, end: str, out_file: Path):
    """Extract segment via ffmpeg."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['ffmpeg','-y','-i',str(in_file),'-ss',start,'-to',end,str(out_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )


def concat_audios(parts: list, out_file: Path):
    """Concatenate WAV parts into one file."""
    list_txt = out_file.parent/'concat_list.txt'
    with open(list_txt,'w') as f:
        for p in parts: f.write(f"file '{p.resolve()}'\n")
    subprocess.run(
        ['ffmpeg','-y','-f','concat','-safe','0','-i',str(list_txt),'-c','copy',str(out_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    list_txt.unlink()


def extract_duration(audio_file: Path) -> float:
    out = subprocess.check_output([
        'ffprobe','-v','error','-show_entries','format=duration',
        '-of','default=noprint_wrappers=1:nokey=1', str(audio_file)
    ])
    return float(out.strip())


def group_into_sentences(segments: list) -> list:
    """Group word-level segments into sentences."""
    sentences, current = [], []
    for w in segments:
        current.append(w)
        if w['text'].endswith(('.', '!', '?')):
            sentences.append({
                'start': current[0]['start'],
                'end':   current[-1]['end'],
                'text':  ' '.join(x['text'] for x in current),
                'words': current.copy()
            })
            current = []
    if current:
        sentences.append({
            'start': current[0]['start'],
            'end':   current[-1]['end'],
            'text':  ' '.join(x['text'] for x in current),
            'words': current.copy()
        })
    return sentences


def match_sentences(hook_sentences: list, main_segs: list, vid: str) -> pd.DataFrame:
    """Match hook sentences in main word segments."""
    rows = []
    for sent in hook_sentences:
        words = [w['text'] for w in sent['words']]
        n = len(words)
        for i in range(len(main_segs)-n+1):
            window = main_segs[i:i+n]
            if all(window[j]['text']==words[j] for j in range(n)):
                rows.append({
                    'youtube_id': vid,
                    'start_timestamp': f"{window[0]['start']:.3f}",
                    'end_timestamp':   f"{window[-1]['end']:.3f}",
                    'matched_text':    sent['text']
                })
                break
    return pd.DataFrame(rows)


def fuzzy_match_sentences(hook_sentences: list, main_segs: list, vid: str, threshold: int = 90) -> pd.DataFrame:
    """
    Match hook sentences in main transcript with fuzzy matching for more robust results.
    Returns a DataFrame with matched sentences and their timestamps from the main transcript.
    
    Args:
        hook_sentences: List of sentence dictionaries from hook transcript
        main_segs: List of word dictionaries from main transcript
        vid: YouTube video ID
        threshold: Minimum fuzzy matching score (0-100)
        
    Returns:
        DataFrame with columns: youtube_id, start_timestamp, end_timestamp, matched_text, match_score
    """
    rows = []
    
    for sent in hook_sentences:
        sent_text = sent['text']
        sent_word_count = len(sent['words'])
        
        min_window = max(sent_word_count - 2, 3)
        max_window = sent_word_count * 2
        
        best_match = None
        best_score = 0
        
        for window_size in range(min_window, max_window + 1):
            for i in range(len(main_segs) - window_size + 1):
                window = main_segs[i:i+window_size]
                window_text = " ".join(w['text'] for w in window)
                
                score = fuzz.ratio(sent_text.lower(), window_text.lower())
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'start_idx': i,
                        'end_idx': i + window_size - 1,
                        'score': score
                    }
        
        if best_match and best_match['score'] >= threshold:
            start_idx = best_match['start_idx']
            end_idx = best_match['end_idx']
            
            rows.append({
                'youtube_id': vid,
                'start_timestamp': f"{main_segs[start_idx]['start']:.3f}",
                'end_timestamp': f"{main_segs[end_idx]['end']:.3f}",
                'matched_text': sent_text,
                'match_score': best_match['score']
            })
    
    return pd.DataFrame(rows)


def convert_main_to_sentences(main_segs):
    """
    Convert word-level main transcript to sentence-level.
    
    Args:
        main_segs: List of word dictionaries from main transcript
        
    Returns:
        List of sentence dictionaries with start, end, and text
    """
    sentences = []
    current_sentence = []
    
    for word in main_segs:
        current_sentence.append(word)
        
        if word['text'].endswith(('.', '!', '?')):
            if current_sentence:
                sentences.append({
                    'start': current_sentence[0]['start'],
                    'end': current_sentence[-1]['end'],
                    'text': ' '.join(w['text'] for w in current_sentence)
                })
                current_sentence = []
    
    if current_sentence:
        sentences.append({
            'start': current_sentence[0]['start'],
            'end': current_sentence[-1]['end'],
            'text': ' '.join(w['text'] for w in current_sentence)
        })
    
    return sentences


def create_final_transcript(main_segs, hook_matches, vid):
    """
    Create a final transcript file that combines main transcript at sentence level
    with indicators for hook matched sentences.
    
    Args:
        main_segs: List of word dictionaries from main transcript
        hook_matches: DataFrame of matched hook sentences
        vid: YouTube video ID
        
    Returns:
        DataFrame with the complete transcript and hook match indicators
    """
    main_sentences = convert_main_to_sentences(main_segs)
    
    main_df = pd.DataFrame([
        {
            'youtube_id': vid,
            'start_timestamp': f"{s['start']:.3f}",
            'end_timestamp': f"{s['end']:.3f}",
            'text': s['text'],
            'is_hook_match': 0  # Default to 0 (not a hook match)
        }
        for s in main_sentences
    ])
    
    if not hook_matches.empty:
        hook_matches['start_float'] = hook_matches['start_timestamp'].astype(float)
        hook_matches['end_float'] = hook_matches['end_timestamp'].astype(float)
        
        for i, row in main_df.iterrows():
            start = float(row['start_timestamp'])
            end = float(row['end_timestamp'])
            
            for j, hook_row in hook_matches.iterrows():
                hook_start = hook_row['start_float']
                hook_end = hook_row['end_float']
                
                if (start <= hook_end and end >= hook_start):
                    main_df.at[i, 'is_hook_match'] = 1
                    main_df.at[i, 'matched_text'] = hook_row['matched_text']
                    break
    
    if 'start_float' in hook_matches.columns:
        hook_matches = hook_matches.drop(columns=['start_float', 'end_float'])
    
    return main_df

def create_refined_final_transcript(main_segs, hook_matches, vid):
    """
    Create a final transcript with the following approach:
    1. Identify words in the main transcript that correspond to hook matches
    2. Use timestamps from hook_matches for these identified segments
    3. Convert remaining words to sentence-level transcripts
    4. Combine both into a single transcript file with required format
    
    Args:
        main_segs: List of word dictionaries from main transcript
        hook_matches: DataFrame of matched hook sentences with timestamps
        vid: YouTube video ID
        
    Returns:
        DataFrame with the complete transcript in required format
    """
    is_hook_match = [False] * len(main_segs)
    hook_text_mapping = [None] * len(main_segs)
    hook_timestamp_mapping = [None] * len(main_segs)
    
    if not hook_matches.empty:
        hook_matches['start_float'] = hook_matches['start_timestamp'].astype(float)
        hook_matches['end_float'] = hook_matches['end_timestamp'].astype(float)
        
        for _, hook_row in hook_matches.iterrows():
            hook_start = hook_row['start_float']
            hook_end = hook_row['end_float']
            hook_text = hook_row['matched_text']
            
            for i, word in enumerate(main_segs):
                word_start = word['start']
                word_end = word['end']
                
                if (word_start <= hook_end and word_end >= hook_start):
                    is_hook_match[i] = True
                    hook_text_mapping[i] = hook_text
                    hook_timestamp_mapping[i] = (hook_row['start_timestamp'], hook_row['end_timestamp'])
    
    final_segments = []
    
    i = 0
    hook_counter = 0  # Counter for hook_rank
    while i < len(main_segs):
        if is_hook_match[i]:
            start_idx = i
            hook_text = hook_text_mapping[i]
            hook_timestamps = hook_timestamp_mapping[i]
            
            while (i < len(main_segs) and is_hook_match[i] and 
                   hook_text_mapping[i] == hook_text):
                i += 1
            
            hook_counter += 1
            
            final_segments.append({
                'transcript_id': vid,
                'start_timestamp': hook_timestamps[0],
                'end_timestamp': hook_timestamps[1],
                'text': hook_text,
                'is_hook': 1,
                'hook_rank': hook_counter
            })
        else:
            sentence_words = []
            while i < len(main_segs) and not is_hook_match[i]:
                sentence_words.append(main_segs[i])
                
                if (main_segs[i]['text'].endswith(('.', '!', '?')) or 
                    (i+1 < len(main_segs) and is_hook_match[i+1])):
                    
                    if sentence_words:
                        final_segments.append({
                            'transcript_id': vid,
                            'start_timestamp': f"{sentence_words[0]['start']:.3f}",
                            'end_timestamp': f"{sentence_words[-1]['end']:.3f}",
                            'text': ' '.join(w['text'] for w in sentence_words),
                            'is_hook': 0,
                            'hook_rank': -1  # -1 for non-hooks as specified
                        })
                        sentence_words = []
                
                i += 1
            
            if sentence_words:
                final_segments.append({
                    'transcript_id': vid,
                    'start_timestamp': f"{sentence_words[0]['start']:.3f}",
                    'end_timestamp': f"{sentence_words[-1]['end']:.3f}",
                    'text': ' '.join(w['text'] for w in sentence_words),
                    'is_hook': 0,
                    'hook_rank': -1  # -1 for non-hooks as specified
                })
    
    final_df = pd.DataFrame(final_segments)
    
    final_df['start_float'] = final_df['start_timestamp'].astype(float)
    final_df['end_float'] = final_df['end_timestamp'].astype(float)
    
    final_df = final_df.sort_values(['start_float', 'end_float'])
    
    final_df = final_df.drop(columns=['start_float', 'end_float'])
    
    final_df['sentence_id'] = range(1, len(final_df) + 1)
    
    final_df = final_df[['transcript_id', 'sentence_id', 'start_timestamp', 
                          'end_timestamp', 'text', 'is_hook', 'hook_rank']]
    
    return final_df


def main(input_csv: str, work_dir: str='downloads',
         cookies_from_browser: str=None, cookies_file: str=None,
         fuzzy_threshold: int=90):
    """
    Enhanced main function that performs both exact and fuzzy matching.
    Creates a final.csv with the complete transcript in the specified format.
    
    Args:
        input_csv: Path to input CSV file with YouTube URLs and hook timestamps
        work_dir: Directory to store downloaded files and results
        cookies_from_browser: Browser to extract cookies from (for yt-dlp)
        cookies_file: Path to cookies file (for yt-dlp)
        fuzzy_threshold: Threshold for fuzzy matching (0-100)
    """
    df = pd.read_csv(input_csv)
    for url in df['youtube_url'].unique():
        parsed = urlparse(clean_url(url))
        video_id = parse_qs(parsed.query).get('v', [None])[0]
        if not video_id:
            print(f"Warning: Could not extract video ID from URL: {url}, skipping...")
            continue
            
        print(f"Processing video ID: {video_id}")
        
        audio = download_audio(url, Path(work_dir)/'audio', cookies_from_browser, cookies_file)
        
        base = Path(work_dir)/video_id
        (base/'hook_audio').mkdir(parents=True, exist_ok=True)
        (base/'main_audio').mkdir(exist_ok=True)

        rows = df[df['youtube_url']==url]
        start_h, end_h = seconds_to_hhmmss(rows['hook_start'].apply(hhmmss_to_seconds).min()), \
                        seconds_to_hhmmss(rows['hook_end'].apply(hhmmss_to_seconds).max())
        hook_wav = base/'hook_audio'/f"{video_id}_hook.wav"
        cut_segment(audio, start_h, end_h, hook_wav)
        hook_words = transcribe_audio(hook_wav)
        hook_sents = group_into_sentences(hook_words)
        pd.DataFrame([
            {'youtube_id':video_id,'start_timestamp':f"{s['start']:.3f}",
             'end_timestamp':f"{s['end']:.3f}",'sentence_text':s['text']}
            for s in hook_sents
        ]).to_csv(base/'hook_sentences.csv',index=False)

        dur = extract_duration(audio)
        hs, he = hhmmss_to_seconds(start_h), hhmmss_to_seconds(end_h)
        parts=[]
        if hs>0:
            pre=base/'main_audio'/'pre.wav'; cut_segment(audio,'00:00:00',start_h,pre); parts.append(pre)
        if he<dur:
            post=base/'main_audio'/'post.wav'; cut_segment(audio,end_h,seconds_to_hhmmss(dur),post); parts.append(post)
        main_wav=base/'main_audio'/f"{video_id}_main.wav"
        if parts:
            concat_audios(parts, main_wav)
            for p in parts: p.unlink()
        else:
            import shutil
            shutil.copy(audio, main_wav)
        
        main_segs = transcribe_audio(main_wav)
        pd.DataFrame([
            {'start_timestamp':f"{m['start']:.3f}",'end_timestamp':f"{m['end']:.3f}",
             'main_segment_text':m['text']} for m in main_segs
        ]).to_csv(base/'main_transcripts.csv',index=False)

        exact_matched = match_sentences(hook_sents, main_segs, video_id)
        
        fuzzy_matched = fuzzy_match_sentences(hook_sents, main_segs, video_id, fuzzy_threshold)
        
        fuzzy_matched.to_csv(base/'hook_fuzzy_matches.csv', index=False)
        
        high_quality_fuzzy = fuzzy_matched[fuzzy_matched['match_score'] >= fuzzy_threshold]
        
        if not high_quality_fuzzy.empty:
            high_quality_fuzzy = high_quality_fuzzy.drop(columns=['match_score'])
        
        if not exact_matched.empty:
            combined_matches = pd.concat([exact_matched, high_quality_fuzzy]).drop_duplicates()
        else:
            combined_matches = high_quality_fuzzy
        
        combined_matches.to_csv(base/'hook_matched_transcripts.csv', index=False)
        
        final_transcript = create_refined_final_transcript(main_segs, combined_matches, video_id)
        
        print(f"Verifying sort order... First few rows of final transcript:")
        if len(final_transcript) > 0:
            first_few = final_transcript.head(min(5, len(final_transcript)))
            for _, row in first_few.iterrows():
                print(f"  {row['sentence_id']}: {row['start_timestamp']} -> {row['end_timestamp']}")
        
        final_transcript.to_csv(base/'final.csv', index=False)

        hook_count = (final_transcript['is_hook'] == 1).sum()
        total_count = len(final_transcript)

        print(f"Completed {video_id}: outputs in {base}")
        print(f"  - Exact matches: {len(exact_matched)} sentences")
        print(f"  - High-quality fuzzy matches (score ≥ {fuzzy_threshold}): {len(high_quality_fuzzy)} sentences")
        print(f"  - Total matches in hook_matched_transcripts.csv: {len(combined_matches)} sentences")
        print(f"  - Created final.csv with {total_count} segments, including {hook_count} hook matches")


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('input_csv')
    p.add_argument('--work-dir',default='downloads')
    p.add_argument('--cookies-from-browser',default=None)
    p.add_argument('--cookies',dest='cookies_file',default=None)
    p.add_argument('--fuzzy-threshold', type=int, default=90, 
                   help='Threshold for fuzzy matching (0-100)')
    args=p.parse_args()
    
    main(args.input_csv, args.work_dir, args.cookies_from_browser, 
         args.cookies_file, args.fuzzy_threshold)
