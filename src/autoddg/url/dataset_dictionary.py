import os
import re
import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from scipy.optimize import linear_sum_assignment
from .dataset_information import extract_dataset_id

_EMBED_MODEL = None

_SYMBOL_SUBS = [
    ("%", " percent "),
    ("#", " number "),
    ("&", " and "),
    ("@", " at "),
    ("$", " dollar "),
    ("/", " per "),
    ("+", " plus "),
]

_ABBREV_SUBS = [
    (r"\bqty\b", "quantity"),
    (r"\bamt\b", "amount"),
    (r"\bdesc\b", "description"),
    (r"\baddr\b", "address"),
    (r"\bnum\b", "number"),
    (r"\bno\b", "number"),
    (r"\bpct\b", "percent"),
    (r"\bperc\b", "percent"),
    (r"\bpercentage\b", "percent"),
    (r"\byr\b", "year"),
    (r"\bmo\b", "month"),
    (r"\bdob\b", "date of birth"),
    (r"\bdt\b", "date"),
    (r"\bdte\b", "date"),
    (r"\bid\b", "identifier"),
]


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from fastembed import TextEmbedding
        _EMBED_MODEL = TextEmbedding()
    return _EMBED_MODEL


def _embed_matrix(texts):
    model = _get_embed_model()
    arr = np.array(list(model.embed(texts)))
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def get_attachments(url):
    """
    Collecting NYC Open Data attached excel files by examining its URL. 
    
    Arguments: 
    url: url of NYC open data page. 

    Returns: 
    list of tuples: (file name.ext, download link). Contains all attached files on the website, plus their download links. 
    """
    dataset_id = extract_dataset_id(url)
    
    #use dataset endpoint for NYC Open Data API.
    api_url = f"https://data.cityofnewyork.us/api/views/{dataset_id}"
    
    #Get the JSON metadata
    response = requests.get(api_url)
    data = response.json()
    
    #Pull the attachments from the metadata
    attachments = data.get('metadata', {}).get('attachments', [])
        
    downloadable_files = []

    for file in attachments:
        filename = file.get('name')
        asset_id = file.get('assetId')
        
        # Construct the download link
        download_link = f"https://data.cityofnewyork.us/api/views/{dataset_id}/files/{asset_id}"
        downloadable_files.append((filename, download_link))

    return downloadable_files


def normalize(col):
    """
    Normalize a column name for fuzzy/embedding matching.

    Drops text in parenthesis, expands symbols (% -> percent, # -> number etc) 
    and header abbreviationsto their full word forms, then collapses separators 
    to a single underscore so the two sides of a comparison line up token-for-token.
    """
    s = re.sub(r"\(.*?\)", "", str(col)).lower()
    for pat, rep in _SYMBOL_SUBS:
        s = s.replace(pat, rep)
    # treat common separators as spaces so word-boundary regexes apply uniformly
    s = re.sub(r"[_\-.,;:|\\]", " ", s)
    for pat, rep in _ABBREV_SUBS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s


def get_dictionary(files_list, final_df, dataset_shape, threshold=80, embed_threshold=78, min_match_ratio=0.3):
    """
    Check excel/csv files on the dataset website to find data dictionaries. Build input for AutoDDG.

    Pipeline per file:
      1. Open the file as one or more tabular sheets (xlsx -> per sheet; csv -> single sheet).
      2. For each sheet, score how well a small window of its rows aligns with
         the dataset's actual field names. Pick the column inside the sheet that
         holds the column-name labels.
      3. If the alignment is strong enough, harvest every row's metadata
         keyed by whatever label the DD wrote (e.g. "Total Tested (count)").
      4. After all sheets are scanned, remap those raw labels onto the dataset's
         canonical field names using the matches that passed the per-cell gate.

    Arguments:
    files_list: list of tuples (file names.ext, download link)
    final_df: dataframe of information about datasets.
    threshold: per-column match cutoff (0-100, token_sort_ratio).
    embed_threshold: per-column semantic fallback cutoff (cosine * 100).
    min_match_ratio: minimum fraction of dataset columns that must match before a sheet is
        accepted as a data dictionary (guards against tiny sheets with one lucky hit).

    Returns: dict of column names and their additional info, input for autoDDG.
    """
    # input_dict: {raw_DD_label: {metadata_field: value}} — keys are whatever the
    #             DD wrote, before any remapping to canonical field names.
    # match_map:  {normalized_DD_label: canonical_field_name} — built only from
    #             cells that individually passed `threshold`, so we know which
    #             input_dict keys are safe to rename in the final pass.
    input_dict = {}
    match_map = {}
    for file in files_list:
        _, ext = os.path.splitext(file[0])
        ext = ext.lower()

        # Only read excel and csv files 
        if ext in ['.xlsx', '.xls']:
            try:
                xl = pd.ExcelFile(file[1])
            except Exception:
                continue
            sheets = {}
            for sname in xl.sheet_names:
                try:
                    # NYC DDs put a banner row ("Data Dictionary - Column
                    # Information") at row 0 and the real headers at row 1.
                    # header=1 picks up the headers and drops the banner.
                    sheets[sname] = xl.parse(sname, header=1)
                except Exception:
                    pass
        elif ext == '.csv':
            # CSVs are single-table; wrap so the per-sheet loop below stays uniform.
            sheets = {file[0]: pd.read_csv(file[1], header=1)}
        else:
            continue

        for df in sheets.values():
            # Normalize the header row so downstream metadata keys are stable
            df.columns = df.columns.map(normalize)

            # Cap rows scanned to stay cheap even if the file is the full dataset.
            # Allow a small margin over dataset_shape[1] so DDs that document extra
            # columns still fit inside the window.
            n_cols = dataset_shape[1]
            cap = min(n_cols * 2, n_cols + 10)

            # Keep both forms of the dataset's field names: `_orig` for
            # inserting the canonical name into match_map, `_norm` so the
            # comparison is fair against the normalized DD cells.
            final_cols_orig = pd.Series(final_df['columns_field_name'].values[0]).astype(str)
            final_cols_norm = final_cols_orig.map(normalize)

            # The "column name" column isn't always the first one, so we score each 
            # candidate column against the dataset columns and use whichever lines up best.
            n_candidates = min(3, df.shape[1])
            best_pick = None 
            for cand_idx in range(n_candidates):
                cand_cols_norm = df.iloc[:cap, cand_idx].astype(str).map(normalize)
                if len(cand_cols_norm) == 0:
                    continue
                # Lexical similarity for the first pass
                cand_mat = process.cdist(cand_cols_norm, final_cols_norm, scorer=fuzz.token_sort_ratio).astype(float)
                cand_top = cand_mat.max(axis=1)

                # Divide by n_cols so sparse sheets with a couple of lucky hits 
                # don't outrank a candidate that covers the dataset.
                agg = float(np.sum(np.sort(cand_top)[::-1][:n_cols]) / n_cols)
                if best_pick is None or agg > best_pick[3]:
                    best_pick = (cand_idx, cand_cols_norm, cand_mat, agg)

            if best_pick is None:
                continue

            key_col_idx, file_cols_norm, combined, _ = best_pick

            # Semantic fallback: To salvage DDs that paraphrase rather than repeat 
            # the field name, we use embedding similarity as a second pass
            try:
                if (combined.max(axis=1) < threshold).any():
                    file_emb = _embed_matrix(file_cols_norm.tolist())
                    final_emb = _embed_matrix(final_cols_norm.tolist())
                    emb_mat = (file_emb @ final_emb.T) * 100.0

                    # For robustiness, only override where the embedding score is 
                    # both above its own threshhold AND strictly better than the lexical score
                    better = (emb_mat >= embed_threshold) & (emb_mat > combined)
                    combined = np.where(better, emb_mat, combined)
            except Exception as exc:

                print(f"[get_dictionary] embedding fallback skipped: {type(exc).__name__}: {exc}")


            # Each DD row gets matched to at most one dataset column
            row_ind, col_ind = linear_sum_assignment(-combined)
            assigned_scores = combined[row_ind, col_ind]

            # Sheet-level acceptance criteria has two parts:
            # mean assigned score > threshold (overall fit is good), AND
            # at least min_match_ratio of dataset columns individually pass.
            best_score = float(np.sum(assigned_scores) / n_cols)
            n_strong = int(np.sum(assigned_scores >= threshold))

            if best_score > threshold and n_strong >= max(1, int(np.ceil(min_match_ratio * n_cols))):
                # Only use match_map and input_dict from sheets that pass
                for r, c, s in zip(row_ind, col_ind, assigned_scores):
                    if s >= threshold:
                        # Map normalized DD label -> canonical field name.
                        match_map[file_cols_norm.iloc[int(r)]] = final_cols_orig.iloc[int(c)]

                # Use every documented row, not just rows used in scoring
                key_col = df.columns[key_col_idx]
                df = df[df[key_col].notna()]
                for _, row in df.iterrows():
                    key = str(row[key_col]).strip()
                    # Skip DD name column
                    input_dict[key] = {
                        col: row[col]
                        for col in df.columns
                        if col != key_col and pd.notna(row[col]) and str(row[col]).strip() != ""
                    }

    # Rename raw DD labels into canonical field names
    remapped_dict = {}
    for key, value in input_dict.items():
        new_key = match_map.get(normalize(key))
        if new_key is None:
            continue
        remapped_dict[new_key] = value

    return remapped_dict

