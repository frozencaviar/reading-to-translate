"""
Reproduce the analysis and plots from result-analysis.ipynb for YiyuZhang_FIX data.
Closely follows the notebook's methodology with cross-validated OLS regression,
permutation tests, and delta log-likelihood pointplots.
"""
import argparse
import json
import pandas as pd
import numpy as np
import copy, math, os, warnings
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore, pearsonr
from wordfreq import word_frequency
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.unicode_minus'] = False

# ─── Configuration ───────────────────────────────────────────────
DATA_DIR = "attention-norm-yiyu-positive-dur"
FIGURES_DIR = "figures_yiyu_positive_dur"
os.makedirs(FIGURES_DIR, exist_ok=True)
DUR_MIN_MS = 20
DUR_POLICY = "positive_Dur_only_ge_20ms"

sides = ['src', 'tgt']
levels = ['token', 'segment']
side2labels = {'tgt': ['log_TrtT', 'log_Dur'], 'src': ['log_TrtS']}
MAIN_N_SPLITS = 10
GROUP_N_SPLITS = 3
STABILITY_REPEATS = 30
BOOTSTRAP_N = 5000
PERMUTATION_N = 1000
# coef_bar mini-panel aspect ratio: width:height = 5:9
COEF_AX_BOX_ASPECT = 9 / 5  # matplotlib expects height / width
COEF_RIGHT_MARGIN = 0.975
COEF_TOP_MARGIN = 0.82
COEF_LEGEND_Y = 1.00
COEF_EXTRA_RIGHT_INCH = 0.35  # extra whitespace on the right side of the whole coef_bar figure
COEF_EXTRA_SIDE_INCH_LEGEND = 0.45  # symmetric extra canvas on both sides for the legend-bearing panel
MAINDELTA_POINT_COLOR = '#454545'
MAINDELTA_POINT_SIZE = 42
MAINDELTA_TOKEN_MARKER = 'o'
MAINDELTA_SEG_MARKER = '^'
MAINDELTA_DODGE = 0.11
MAINDELTA_PANEL_FIGSIZE = (3.45, 2.85)
MAINDELTA_XLABEL_SIZE = 13
MAINDELTA_TITLE_SIZE = 13
MAINDELTA_YTICK_LABELSIZE = 12
MAINDELTA_ZERO_LINE_COLOR = '#d98b2b'
MAINDELTA_ZERO_LINE_WIDTH = 1.0
MAINDELTA_LEGEND_LOC = 'upper right'
MAINDELTA_LEGEND_FONTSIZE = 9
INTERACTION_LEGEND_Y = -0.06
ATT_LEGEND_Y = 0.0
GROUP_LEGEND_Y = 0.0
ATT_YTICK_LABELSIZE = 9
GROUP_YTICK_LABELSIZE = 9
ATT_ROW_LABEL_RIGHT_X = 1.05
GROUP_ROW_LABEL_RIGHT_X = 1.05
ATT_MAIN_HSPACE = 0.09
GROUP_MAIN_HSPACE = 0.13
GROUP_SPLIT_SEPARATOR_RATIO = 0.20
GROUP_MAIN_MANUAL_YLIM = (-0.05, 0.05)
GROUP_APPENDIX_SPECIAL_YLIMS = {
    'translating_en-zh': (-0.10, 0.09),
    'translating_zh-en': (-0.10, 0.07),
    'post-editing_en-zh': (-0.02, 0.035),
    'post-editing_zh-en': (-1.40, 0.5),
}
ATT_APPENDIX_SPECIAL_YLIMS = {
    'post-editing_zh-en': (-1.4, 0.5),
}
SPLIT_SEPARATOR_RATIO = 0.28
ATT_APPENDIX_LEGEND_Y = -0.025
GROUP_APPENDIX_LEGEND_Y = -0.025
SHRINK_FIGSIZE = (9.4, 5.2)
SHRINK_LINE_COLOR = '#6b7280'
SHRINK_LINE_WIDTH = 0.95
SHRINK_CTRL_COLOR = '#5EA0C7'
SHRINK_SURP_COLOR = '#6BC179'
SHRINK_CTRL_MEAN_COLOR = '#5EA0C7'
SHRINK_SURP_MEAN_COLOR = '#6BC179'
SHRINK_CTRL_MEAN_EDGE_COLOR = '#2F6E93'
SHRINK_SURP_MEAN_EDGE_COLOR = '#3E8B4A'
SHRINK_POINT_SIZE = 20
SHRINK_RAIN_POINT_SIZE = 13
SHRINK_RAIN_POINT_ALPHA = 0.58
SHRINK_Y_STEP = 0.78
SHRINK_CLOUD_Y_OFFSET = 0.05
SHRINK_CLOUD_WIDTH = 0.50
SHRINK_RAIN_Y_OFFSET = 0.12
SHRINK_RAIN_Y_JITTER = 0.012
SHRINK_LABELSIZE = 10.5
SHRINK_LEGEND_LOC = 'lower right'
SHRINK_LEGEND_ANCHOR = (0.98, 0.05)
SHRINK_BAND_COLOR = '#f5f6f8'
SHRINK_ZERO_LINE_COLOR = '#9ca3af'
SHRINK_GRID_COLOR = '#e5e7eb'
SHRINK_SPINE_COLOR = '#4b5563'
SHRINK_DELTA_COLOR = '#374151'
SHRINK_DELTA_SIZE = 8.6
SHRINK_TITLE_COLOR = '#1f2937'
SHRINK_TITLE_SIZE = 11.5
SHRINK_MEAN_EDGE_WIDTH = 0.55
POS_PRIMARY_COLOR = '#14283a'
POS_SECONDARY_COLOR = '#3e566f'
POS_ERR_COLOR = '#4f5c68'
POS_TITLE_SIZE = 12
POS_LABEL_SIZE = 12
POS_TICK_SIZE = 11
POS_SPINE_COLOR = 'black'
POS_SPINE_WIDTH = 1.0
TGT_POS_PANEL_BOX_ASPECT = 1.5
EXAMPLE_PANEL_BOX_ASPECT = 1.5
EXAMPLE_LINE_COLOR = '#6b7682'
EXAMPLE_TITLE_SIZE = 12
EXAMPLE_TOKEN_SIZE = 10
SIG_STAR_SIZE = 12
SIG_STAR_WHITE_COLOR = 'white'
SIG_STAR_BLACK_COLOR = 'black'
PLOT_KEYS = {
    'main_delta',
    'coef_bars',
    'att_delta',
    'att_delta_appendix',
    'att_shrinkage',
    'vif',
    'group_delta',
    'group_delta_appendix',
    'interaction_task',
    'interaction_direction',
    'src_pos',
    'tgt_pos',
    'example_sentence',
}
INTERACTION_FAMILY_MAP = {
    'task': 'task_int',
    'direction': 'direction_int',
}

features = {
    'src': ['src_surp', 'x2x', 'x2cont', 'x2eos', 'x_ent', 'cont2x', 'tgtseq2x'],
    'tgt': ['tgt_surp', 'xent', 'eos_xatt', 'y2srcseq_ent', 'y2y', 'y2cont', 'y_ent']
}
attention_features = {
    'src': ['x2x', 'x2cont', 'x2eos', 'x_ent', 'cont2x', 'tgtseq2x'],
    'tgt': ['eos_xatt', 'y2srcseq_ent', 'y2y', 'y2cont', 'y_ent']
}

control_features = {
    side: {level: [f'{side}_seg_len', f'{side}_freq', f'{side}_posq'] for level in levels}
    for side in sides
}
for side in sides:
    control_features[side]['sentence'] = [f'{side}_seg_len', f'{side}_freq']

feature_names = {
    'y2srcseq_ent': r'$H^\mathrm{c}_{v,x}$',
    'eos_xatt': r'$f^\mathrm{c}_{v,\mathrm{eos}}$',
    'xent': r'$S_{\mathrm{mt}}(v)$',
    'tgt_surp': r'$S_{\mathrm{lm}}(v)$',
    'y_ent': r'$H^\mathrm{d}_{v, \overleftarrow{v}}$',
    'src_surp': r'$S_{\mathrm{lm}}(u)$',
    'x2x': r'$f^\mathrm{e}_{u,u}$',
    'x2cont': r'$f^\mathrm{e}_{u,\bar{u}}$',
    'x2eos': r'$f^\mathrm{e}_{u,\mathrm{eos}}$',
    'y2y': r'$f^\mathrm{d}_{v,v}$',
    'y2cont': r'$f^\mathrm{d}_{v,\bar{v}}$',
    'cont2x': r'$f^\mathrm{e}_{\bar{u},u}$',
    'x_ent': r'$H^\mathrm{e}_{u,x}$',
    'tgtseq2x': r'$f^\mathrm{c}_{y,u}$',
}

rng = np.random.default_rng(42)


@dataclass
class SigResult:
    pvalue: float
    n_participants: int

GROUP_LABELS = [
    'all',
    'all_translating',
    'all_post-editing',
    'all_en-zh',
    'all_zh-en',
    'translating_en-zh',
    'translating_zh-en',
    'post-editing_en-zh',
    'post-editing_zh-en',
]

GROUP_FILTERS = {
    'all': {},
    'all_translating': {'task_type': 'translating'},
    'all_post-editing': {'task_type': 'post-editing'},
    'all_en-zh': {'lang_pair': 'en-zh'},
    'all_zh-en': {'lang_pair': 'zh-en'},
    'translating_en-zh': {'lang_pair': 'en-zh', 'task_type': 'translating'},
    'translating_zh-en': {'lang_pair': 'zh-en', 'task_type': 'translating'},
    'post-editing_en-zh': {'lang_pair': 'en-zh', 'task_type': 'post-editing'},
    'post-editing_zh-en': {'lang_pair': 'zh-en', 'task_type': 'post-editing'},
}

MAIN_SUMMARY_GROUPS = [
    'all',
    'all_translating',
    'all_post-editing',
    'all_en-zh',
    'all_zh-en',
]

FULL_SPLIT_GROUPS = [
    'translating_en-zh',
    'translating_zh-en',
    'post-editing_en-zh',
    'post-editing_zh-en',
]

MAIN_ROW_LAYOUT = [
    'all',
    'all_translating',
    'all_post-editing',
    None,
    'all_en-zh',
    'all_zh-en',
]

APPENDIX_ROW_LAYOUT = [
    'translating_en-zh',
    'translating_zh-en',
    None,
    'post-editing_en-zh',
    'post-editing_zh-en',
]

GROUP_DISPLAY_NAMES = {
    'all': 'All',
    'all_translating': 'All\nHT',
    'all_post-editing': 'All\nPE',
    'all_en-zh': 'All\nen->zh',
    'all_zh-en': 'All\nzh->en',
    'translating_en-zh': 'HT\nen->zh',
    'translating_zh-en': 'HT\nzh->en',
    'post-editing_en-zh': 'PE\nen->zh',
    'post-editing_zh-en': 'PE\nzh->en',
}

TARGET_TO_SIDE = {
    'log_TrtS': 'src',
    'log_TrtT': 'tgt',
    'log_Dur': 'tgt',
}

# ─── Data Loading & Preprocessing ────────────────────────────────

def get_word_frequency(words, lang):
    mean_fq = np.mean([max(word_frequency(w, lang), 1e-8) for w in str(words).split('_')])
    return np.log(mean_fq)


def infer_task_type(session_id):
    sid = str(session_id)
    parts = sid.split('_')
    if len(parts) > 1 and parts[1]:
        task_code = parts[1][0].upper()
        return {'T': 'translating', 'P': 'post-editing', 'C': 'copying'}.get(task_code, 'other')
    return 'other'

def infer_task_code(session_id):
    sid = str(session_id)
    parts = sid.split('_')
    return parts[1] if len(parts) > 1 and parts[1] else 'unknown'

def preprocess(df, standardize=True):
    # Stable row-level id used to align per-fold predictions.
    df = df.copy()
    df["row_uid"] = np.arange(len(df), dtype=int)

    for label in ['TrtS', 'Dur', 'TrtT']:
        if label in df.columns:
            df[label] = pd.to_numeric(df[label], errors='coerce')

    df['valid_TrtS_ge20'] = df['TrtS'] >= DUR_MIN_MS
    df['valid_TrtT_ge20'] = df['TrtT'] >= DUR_MIN_MS
    df['valid_Dur_positive_ge20'] = (df['Dur'] > 0) & (df['Dur'] >= DUR_MIN_MS)

    df['log_TrtS'] = np.where(df['valid_TrtS_ge20'], np.log(df['TrtS']), np.nan)
    df['log_TrtT'] = np.where(df['valid_TrtT_ge20'], np.log(df['TrtT']), np.nan)
    # Dur is modeled only on the subset with actual positive production time.
    df['log_Dur'] = np.where(df['valid_Dur_positive_ge20'], np.log(df['Dur']), np.nan)
    df['dur_policy'] = DUR_POLICY

    if standardize:
        for label in ['log_TrtS', 'log_TrtT', 'log_Dur']:
            df[label] = zscore(df[label].values, nan_policy='omit')

    for side in sides:
        df[f'{side}_tokens'] = df[f'{side}_tokens'].astype(str)
        df[f'{side}_freq'] = df[[f'{side}_tokens', f'{side}_lang']].apply(
            lambda x: get_word_frequency(x.iloc[0], x.iloc[1]), axis=1)

    df["lang_pair"] = df[["src_lang", "tgt_lang"]].apply(lambda x: '-'.join(x), axis=1)
    df["task_type"] = df["session_id"].apply(infer_task_type)
    df["task_code"] = df["session_id"].apply(infer_task_code)
    df["src_seq_id"] = df[["study", "session_id", 'src_segid']].apply(
        lambda x: f"{x.iloc[0]}_{x.iloc[1]}_{x.iloc[2]}", axis=1)
    df["material_id"] = df[["study", "task_code"]].apply(
        lambda x: f"{x.iloc[0]}::{x.iloc[1]}", axis=1)
    df["sentence_item_id"] = df[["material_id", "src_segid"]].apply(
        lambda x: f"{x.iloc[0]}::s{x.iloc[1]}", axis=1)
    df["src_unit_item_id"] = df[["sentence_item_id", "src_token_ids"]].apply(
        lambda x: f"{x.iloc[0]}::src_{x.iloc[1]}", axis=1)
    df["tgt_position_item_id"] = df[["sentence_item_id", "tgt_token_ids"]].apply(
        lambda x: f"{x.iloc[0]}::tgt_{x.iloc[1]}", axis=1)
    # The source-aligned unit is the preferred item id for mixed-effects models.
    # It is stable across translators, unlike target positions in free translation.
    df["item_id"] = df["src_unit_item_id"]
    df["session_sentence_id"] = df["src_seq_id"]
    # Existing cross-validation key. This remains session-specific to preserve
    # the current analysis pipeline; use sentence_item_id for item-held-out CV.
    df["sentence_id"] = df["src_seq_id"]
    df = df.astype({"tgt_token_ids": "str"})
    df["src_seg_len"] = [len(ids.split("+")) for ids in df.src_token_ids]
    df["tgt_seg_len"] = df["tgt_token_ids"].apply(
        lambda ids: len(ids.split("+")) if pd.notnull(ids) else 0)
    fallback_translator = df[['study', 'session_id']].apply(
        lambda x: '{}_{}'.format(x.iloc[0], x.iloc[1].split('_')[0]), axis=1)
    if 'PartID' in df.columns:
        part_id = df['PartID'].astype(str).str.strip()
        part_id = part_id.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        df["translator_id"] = part_id.fillna(fallback_translator)
    else:
        df["translator_id"] = fallback_translator
    df['y_ent'] = df['y_ent'].fillna(0)

    if standardize:
        for ft in ['src_posq', 'tgt_posq', 'xent', 'xmi', 'x2cont', 'cont2x', 'x_ent',
                    'x2eos', 'tgtseq2x', 'src_surp', 'y2srcseq_ent', 'eos_xatt',
                    'y2cont', 'y_ent', 'tgt_surp', 'src_freq', 'tgt_freq', 'x2x', 'y2y']:
            if ft in df.columns:
                df[ft] = pd.to_numeric(df[ft], errors='coerce')
                df[ft] = zscore(df[ft].values, nan_policy='omit')

    return df


def get_data(att_dir, standardize=True):
    tok_seg_df = preprocess(pd.read_csv(f'{att_dir}/yiyu-features-token-segment.tsv', sep='\t'), standardize)
    seg_tok_df = preprocess(pd.read_csv(f'{att_dir}/yiyu-features-segment-token.tsv', sep='\t'), standardize)
    sent_sent_df = preprocess(pd.read_csv(f'{att_dir}/yiyu-features-sentence-sentence.tsv', sep='\t'), standardize)

    data = {
        'src': {
            'token': tok_seg_df,
            'segment': seg_tok_df.drop_duplicates(["study", "session_id", 'src_segid', 'src_token_ids']),
            'sentence': sent_sent_df
        },
        'tgt': {
            'token': seg_tok_df,
            'segment': tok_seg_df.drop_duplicates(["study", "session_id", 'src_segid', 'tgt_token_ids']),
            'sentence': sent_sent_df
        }
    }
    return data


def build_preprocessing_summary(data):
    records = []
    for side in sides:
        for level, df in data[side].items():
            for label in side2labels[side]:
                valid_mask = df[label].notnull() if label in df.columns else pd.Series(False, index=df.index)
                records.append({
                    'side': side,
                    'level': level,
                    'label': label,
                    'dur_policy': DUR_POLICY if label == 'log_Dur' else '',
                    'total_rows': int(len(df)),
                    'valid_rows': int(valid_mask.sum()),
                    'dropped_rows': int((~valid_mask).sum()),
                    'valid_rate': float(valid_mask.mean()) if len(df) else np.nan,
                    'unique_sessions': int(df['session_id'].nunique()) if 'session_id' in df.columns else np.nan,
                    'unique_sentences': int(df['sentence_id'].nunique()) if 'sentence_id' in df.columns else np.nan,
                    'unique_translators': int(df['translator_id'].nunique()) if 'translator_id' in df.columns else np.nan,
                    'n_translating': int((df['task_type'] == 'translating').sum()) if 'task_type' in df.columns else np.nan,
                    'n_post_editing': int((df['task_type'] == 'post-editing').sum()) if 'task_type' in df.columns else np.nan,
                })
    return pd.DataFrame(records)


def filter_group_data(data, group_label):
    filters = GROUP_FILTERS[group_label]
    lang_pair = filters.get('lang_pair')
    task_type = filters.get('task_type')

    filtered = {}
    for side in sides:
        filtered[side] = {}
        for level, df in data[side].items():
            curr = df
            if lang_pair is not None and 'lang_pair' in curr.columns:
                curr = curr[curr['lang_pair'] == lang_pair]
            if task_type is not None and 'task_type' in curr.columns:
                curr = curr[curr['task_type'] == task_type]
            filtered[side][level] = curr.copy()
    return filtered


# ─── OLS Model ───────────────────────────────────────────────────

def likelihood(preds, labels, std):
    """Log-likelihood of predictions given labels and std."""
    ll = 0
    for p, l in zip(preds, labels):
        ll += -0.5 * math.log(2 * math.pi * std**2) - (p - l)**2 / (2 * std**2)
    return ll


def fit_formula_model(train, test, formula, label, required_cols):
    train = train[train[label].notnull()].copy()
    test = test[test[label].notnull()].copy()
    if len(train) < 20 or len(test) < 5:
        return None

    numeric_cols = [
        c for c in required_cols
        if c in train.columns and c in test.columns and c not in {label, 'task_type', 'lang_pair'}
    ]
    for col in numeric_cols + [label]:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        test[col] = pd.to_numeric(test[col], errors='coerce')

    train = train.dropna(subset=required_cols + [label])
    test = test.dropna(subset=[c for c in required_cols if c in test.columns] + [label])
    if len(train) < 20 or len(test) < 5:
        return None

    try:
        mdf = smf.ols(formula, data=train).fit()
    except Exception:
        return None

    train_preds = np.asarray(mdf.predict(train), dtype=float)
    test_preds = np.asarray(mdf.predict(test), dtype=float)
    std = np.std(train_preds - train[label].values)
    if std < 1e-10:
        std = 1e-10
    llh = [likelihood([pred], [lab], std) for pred, lab in zip(test_preds, test[label].values)]
    meta_cols = ['row_uid', 'translator_id', 'sentence_id', 'session_id', 'task_type', 'lang_pair']
    meta_cols = [c for c in meta_cols if c in test.columns]
    test_meta = test[meta_cols].copy()
    test_meta['label_value'] = test[label].values
    test_meta['pred_value'] = test_preds
    test_meta['llh'] = llh
    coef_source = getattr(mdf, "params", pd.Series(dtype=float))
    params = pd.Series({ft: float(coef_source.get(ft, np.nan)) for ft in numeric_cols})
    return {'test_llh': llh, 'params': params, 'md': mdf, 'test_meta': test_meta}


def linear_regression(train, test, feats, label):
    formula = f"{label} ~ " + " + ".join(feats + ["C(lang_pair)"])
    required_cols = feats + ['lang_pair']
    return fit_formula_model(train, test, formula, label, required_cols)


def linear_regression_interaction(train, test, ctrl_feats, predictor, label, interaction='task'):
    additive_terms = ctrl_feats + [predictor]
    additive_formula = f"{label} ~ " + " + ".join(additive_terms + ["C(task_type)", "C(lang_pair)"])
    if interaction == 'task':
        interaction_formula = (
            f"{label} ~ " + " + ".join(ctrl_feats + [f"{predictor} * C(task_type)", "C(lang_pair)"])
        )
    elif interaction == 'direction':
        interaction_formula = (
            f"{label} ~ " + " + ".join(ctrl_feats + [f"{predictor} * C(lang_pair)", "C(task_type)"])
        )
    else:
        raise ValueError(f"Unsupported interaction type: {interaction}")

    required_cols = ctrl_feats + [predictor, 'task_type', 'lang_pair']
    additive_res = fit_formula_model(train, test, additive_formula, label, required_cols)
    if additive_res is None:
        return None
    interaction_res = fit_formula_model(train, test, interaction_formula, label, required_cols)
    if interaction_res is None:
        return None

    add_meta = additive_res.get('test_meta', pd.DataFrame()).copy()
    int_meta = interaction_res.get('test_meta', pd.DataFrame()).copy()
    if add_meta.empty or int_meta.empty or 'row_uid' not in add_meta.columns or 'row_uid' not in int_meta.columns:
        interaction_res['test_llh_delta'] = [llh - base_llh for llh, base_llh
                                             in zip(interaction_res['test_llh'], additive_res['test_llh'])]
        interaction_res['test_llh_delta_df'] = pd.DataFrame({'delta': interaction_res['test_llh_delta']})
        return interaction_res

    add_cols = ['row_uid', 'llh'] + [c for c in ['translator_id', 'sentence_id', 'session_id', 'task_type', 'lang_pair'] if c in add_meta.columns]
    int_cols = ['row_uid', 'llh'] + [c for c in ['translator_id', 'sentence_id', 'session_id', 'task_type', 'lang_pair'] if c in int_meta.columns]
    add_view = add_meta[add_cols].rename(columns={'llh': 'llh_add'})
    int_view = int_meta[int_cols].rename(columns={'llh': 'llh_int'})
    merged = int_view.merge(add_view[['row_uid', 'llh_add']], on='row_uid', how='inner')
    merged['delta'] = merged['llh_int'] - merged['llh_add']
    interaction_res['test_llh_delta'] = merged['delta'].tolist()
    keep_cols = ['row_uid', 'delta'] + [c for c in ['translator_id', 'sentence_id', 'session_id', 'task_type', 'lang_pair'] if c in merged.columns]
    interaction_res['test_llh_delta_df'] = merged[keep_cols].copy()
    return interaction_res


def get_train_test_split(data, n_splits=10, random_seed=42):
    """Create train/test splits by sentence_id."""
    trains, tests = [], []
    all_sentences = sorted(data['src']['token']['sentence_id'].unique())
    if len(all_sentences) < 2:
        return trains, tests

    n_splits = min(n_splits, len(all_sentences))
    np.random.seed(random_seed)
    np.random.shuffle(all_sentences)
    folds = np.array_split(np.array(all_sentences), n_splits)

    for fold in folds:
        test_sentences = set(fold.tolist())
        if not test_sentences:
            continue
        train_data = {side: {level: data[side][level][~data[side][level]['sentence_id'].isin(test_sentences)]
                             for level in data[side]} for side in sides}
        test_data = {side: {level: data[side][level][data[side][level]['sentence_id'].isin(test_sentences)]
                            for level in data[side]} for side in sides}
        trains.append(train_data)
        tests.append(test_data)
    return trains, tests


def update_results(all_results, results):
    for side in sides:
        for level in levels:
            for label in side2labels[side]:
                if label not in results[side][level]:
                    continue
                for gp, res in results[side][level][label].items():
                    if gp not in all_results[side][level][label]:
                        all_results[side][level][label][gp] = []
                    all_results[side][level][label][gp].append(res)
    return all_results


def cross_validate(trains, tests, ctrl_features, feats):
    all_results = {side: {level: {label: {} for label in side2labels[side]}
                          for level in levels} for side in sides}
    for train, test in zip(trains, tests):
        results = {side: {level: {label: {} for label in side2labels[side]}
                          for level in levels} for side in sides}
        for side in sides:
            for level in levels:
                for label in side2labels[side]:
                    gp_result = {}
                    gp_result['control'] = linear_regression(
                        train[side][level], test[side][level],
                        ctrl_features[side][level], label)
                    if gp_result['control'] is None:
                        continue
                    for gp in feats[side]:
                        currset = f'control+{gp}'
                        res = linear_regression(
                            train[side][level], test[side][level],
                            ctrl_features[side][level] + [gp], label)
                        if res is None:
                            continue
                        ctrl_meta = gp_result['control'].get('test_meta', pd.DataFrame()).copy()
                        pred_meta = res.get('test_meta', pd.DataFrame()).copy()
                        if ctrl_meta.empty or pred_meta.empty or 'row_uid' not in ctrl_meta.columns or 'row_uid' not in pred_meta.columns:
                            # Fallback for unexpected missing metadata.
                            res['test_llh_delta'] = [llh - ctrl_llh for llh, ctrl_llh
                                                     in zip(res['test_llh'], gp_result['control']['test_llh'])]
                            res['test_llh_delta_df'] = pd.DataFrame({'delta': res['test_llh_delta']})
                        else:
                            ctrl_cols = ['row_uid', 'llh'] + [c for c in ['translator_id', 'sentence_id', 'session_id', 'task_type', 'lang_pair'] if c in ctrl_meta.columns]
                            pred_cols = ['row_uid', 'llh'] + [c for c in ['translator_id', 'sentence_id', 'session_id', 'task_type', 'lang_pair'] if c in pred_meta.columns]
                            ctrl_view = ctrl_meta[ctrl_cols].rename(columns={'llh': 'llh_ctrl'})
                            pred_view = pred_meta[pred_cols].rename(columns={'llh': 'llh_pred'})
                            merged = pred_view.merge(
                                ctrl_view[['row_uid', 'llh_ctrl']],
                                on='row_uid',
                                how='inner'
                            )
                            merged['delta'] = merged['llh_pred'] - merged['llh_ctrl']
                            res['test_llh_delta'] = merged['delta'].tolist()
                            keep_cols = ['row_uid', 'delta'] + [c for c in ['translator_id', 'sentence_id', 'session_id', 'task_type', 'lang_pair'] if c in merged.columns]
                            res['test_llh_delta_df'] = merged[keep_cols].copy()
                        gp_result[currset] = res
                    results[side][level][label] = gp_result
        all_results = update_results(all_results, results)
    return all_results


def cross_validate_interaction(trains, tests, ctrl_features, feats, interaction='task'):
    all_results = {side: {level: {label: {} for label in side2labels[side]}
                          for level in levels} for side in sides}
    for train, test in zip(trains, tests):
        results = {side: {level: {label: {} for label in side2labels[side]}
                          for level in levels} for side in sides}
        for side in sides:
            for level in levels:
                for label in side2labels[side]:
                    for gp in feats[side]:
                        res = linear_regression_interaction(
                            train[side][level], test[side][level],
                            ctrl_features[side][level], gp, label, interaction=interaction
                        )
                        if res is None:
                            continue
                        currset = f'control+{gp}'
                        results[side][level][label][currset] = res
        all_results = update_results(all_results, results)
    return all_results


# ─── Metric Extraction ───────────────────────────────────────────

def get_metric_vals(all_results, metric, concat_values=False):
    vals = {side: {level: {label: {} for label in side2labels[side]}
                   for level in levels} for side in sides}
    for side in sides:
        for level in levels:
            for label in side2labels[side]:
                gps, vls = [], []
                for gp, results_list in all_results[side][level][label].items():
                    if gp == 'control':
                        continue
                    if concat_values:
                        metric_vals = []
                        for res in results_list:
                            if res is not None and metric in res:
                                metric_vals.extend(res[metric])
                        gps.append(gp)
                        vls.append(np.mean(metric_vals) if metric_vals else 0)
                    else:
                        for res in results_list:
                            if res is not None and metric in res:
                                gps.append(gp)
                                vls.append(np.mean(res[metric]))
                vals[side][level][label] = pd.DataFrame({'predictor': gps, 'metric': vls})
    return vals


def signflip_permutation_pvalue(values, n_resamples=PERMUTATION_N, rng_obj=None):
    """Two-sided sign-flip permutation test on participant-level means."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return np.nan
    if rng_obj is None:
        rng_obj = np.random.default_rng(42)
    observed = float(np.mean(vals))
    if n == 1:
        return 1.0
    perm_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        signs = rng_obj.choice([-1.0, 1.0], size=n)
        perm_stats[i] = np.mean(vals * signs)
    pval = (np.sum(np.abs(perm_stats) >= abs(observed)) + 1.0) / (n_resamples + 1.0)
    return float(pval)


def participant_level_feature_values(results_list, metric='test_llh_delta_df'):
    """Aggregate fold-level deltas to participant-level means."""
    frames = []
    for res in results_list:
        if res is None:
            continue
        df_delta = res.get(metric)
        if isinstance(df_delta, pd.DataFrame) and not df_delta.empty and 'translator_id' in df_delta.columns and 'delta' in df_delta.columns:
            frames.append(df_delta[['translator_id', 'delta']].copy())
    if not frames:
        return np.array([], dtype=float), 0
    joined = pd.concat(frames, ignore_index=True)
    per_participant = joined.groupby('translator_id', as_index=False)['delta'].mean()
    vals = per_participant['delta'].to_numpy(dtype=float)
    return vals, int(len(per_participant))


def get_delta_sigfig(all_results, metric='test_llh_delta_df'):
    sigfig = {side: {level: {label: {} for label in side2labels[side]}
                     for level in levels} for side in sides}
    for side in sides:
        for level in levels:
            for label in side2labels[side]:
                for gp, results_list in all_results[side][level][label].items():
                    if gp == 'control':
                        continue
                    ft_name = gp.replace('control+', '')
                    ft_vals, n_parts = participant_level_feature_values(results_list, metric=metric)
                    if len(ft_vals) > 0:
                        p_raw = signflip_permutation_pvalue(
                            ft_vals, n_resamples=PERMUTATION_N, rng_obj=rng
                        )
                        sigfig[side][level][label][ft_name] = SigResult(
                            pvalue=p_raw,
                            n_participants=n_parts
                        )
    return sigfig


def add_surp_delta_alias(results):
    """Alias test_llh_delta as test_llh_delta_surp for surprisal-controlled runs."""
    for side in sides:
        for level in levels:
            for label in side2labels[side]:
                for gp_key, res_list in results[side][level][label].items():
                    if gp_key == 'control':
                        continue
                    for res in res_list:
                        if res is not None and 'test_llh' in res:
                            res['test_llh_delta_surp'] = res.get('test_llh_delta', [])


def compute_dynamic_ylim(values, default=(-0.06, 0.06)):
    """Robust y-limits from values, keeping 0 in view and ignoring extreme outliers."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default

    # Use central quantiles to reduce the effect of one-off outliers.
    lo = np.quantile(arr, 0.10)
    hi = np.quantile(arr, 0.90)
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)

    span = hi - lo
    pad = max(0.004, span * 0.20)
    ymin = lo - pad
    ymax = hi + pad

    # Keep a minimum visible range around zero so rows don't look empty.
    if ymax - ymin < 0.04:
        mid = (ymax + ymin) / 2
        ymin, ymax = mid - 0.02, mid + 0.02

    # Always keep zero in the plot area.
    ymin = min(ymin, -1e-3)
    ymax = max(ymax, 1e-3)
    return ymin, ymax


def benjamini_hochberg_adjust(pvalues):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvalues, dtype=float)
    n = pvals.size
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        bh_val = sorted_p[i] * n / rank
        prev = min(prev, bh_val)
        adjusted[i] = min(prev, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def compute_global_q_lookup(grouped_results, att_features_only, interaction_sigfigs=None):
    """Create global BH-FDR lookup across all reported feature tests."""
    key_to_p = {}
    for group, row_info in grouped_results.items():
        if row_info.get('status') != 'ok':
            continue
        row_sigfig = row_info.get('sigfig', {})
        row_surp_sigfig = row_info.get('surp_sigfig', {})
        for side in sides:
            for level in levels:
                for label in side2labels[side]:
                    # Base control tests (also used by Figure 5 and Figure 1(all))
                    for ft, sig_obj in row_sigfig.get(side, {}).get(level, {}).get(label, {}).items():
                        key = ('ctrl', group, side, label, level, ft)
                        key_to_p[key] = float(sig_obj.pvalue)
                    # Attention tests against control+surprisal (Figure 3)
                    for ft, sig_obj in row_surp_sigfig.get(side, {}).get(level, {}).get(label, {}).items():
                        # Only attention features are plotted in Figure 3.
                        if ft not in att_features_only.get(side, []):
                            continue
                        key = ('ctrl+surp', group, side, label, level, ft)
                        key_to_p[key] = float(sig_obj.pvalue)

    interaction_sigfigs = interaction_sigfigs or {}
    for interaction_type, sigfig_map in interaction_sigfigs.items():
        family = INTERACTION_FAMILY_MAP[interaction_type]
        for side in sides:
            for level in levels:
                for label in side2labels[side]:
                    for ft, sig_obj in sigfig_map.get(side, {}).get(level, {}).get(label, {}).items():
                        key = (family, 'all', side, label, level, ft)
                        key_to_p[key] = float(sig_obj.pvalue)

    keys = list(key_to_p.keys())
    pvals = [key_to_p[k] for k in keys]
    qvals = benjamini_hochberg_adjust(pvals)
    q_lookup = {k: float(q) for k, q in zip(keys, qvals)}
    family_counts = {}
    for key in keys:
        fam = key[0]
        family_counts[fam] = family_counts.get(fam, 0) + 1
    family_counts['m_total'] = len(keys)
    scope_df = pd.DataFrame(
        [{'family': fam, 'n_tests': n} for fam, n in family_counts.items()]
    )
    return q_lookup, scope_df


def build_interaction_delta_table(interaction_delta, interaction_sigfigs, q_lookup):
    records = []
    for interaction_type, delta_map in interaction_delta.items():
        family = INTERACTION_FAMILY_MAP[interaction_type]
        sigfig_map = interaction_sigfigs[interaction_type]
        for side in sides:
            for level in levels:
                for label in side2labels[side]:
                    df_tmp = delta_map[side][level][label].copy()
                    if df_tmp.empty:
                        continue
                    df_tmp['feature'] = df_tmp['predictor'].apply(lambda x: x.split('+')[-1])
                    for _, row in df_tmp.iterrows():
                        ft = row['feature']
                        sig_obj = sigfig_map[side][level][label].get(ft)
                        p_raw = float(sig_obj.pvalue) if sig_obj is not None else np.nan
                        n_participants = int(sig_obj.n_participants) if sig_obj is not None else np.nan
                        q_bh = q_lookup.get((family, 'all', side, label, level, ft), np.nan)
                        records.append({
                            'interaction_type': interaction_type,
                            'side': side,
                            'label': label,
                            'level': level,
                            'predictor': row['predictor'],
                            'feature': ft,
                            'metric': float(row['metric']),
                            'p_raw': p_raw,
                            'q_bh': float(q_bh) if pd.notnull(q_bh) else np.nan,
                            'sig_bh': bool(np.isfinite(q_bh) and q_bh < 0.05),
                            'n_participants': n_participants,
                        })
    return pd.DataFrame(records)


def build_interaction_top_table(interaction_df):
    if interaction_df is None or interaction_df.empty:
        return pd.DataFrame(columns=[
            'interaction_type', 'side', 'target', 'level',
            'top_feature', 'delta_llh_int', 'q_bh', 'sig_bh', 'n_participants'
        ])
    records = []
    for interaction_type in ['task', 'direction']:
        sub = interaction_df[interaction_df['interaction_type'] == interaction_type].copy()
        if sub.empty:
            continue
        for side in sides:
            for level in levels:
                for label in side2labels[side]:
                    cell = sub[(sub['side'] == side) & (sub['level'] == level) & (sub['label'] == label)].copy()
                    if cell.empty:
                        continue
                    cell = cell.sort_values('metric', ascending=False)
                    top = cell.iloc[0]
                    records.append({
                        'interaction_type': interaction_type,
                        'side': side,
                        'target': label,
                        'level': level,
                        'top_feature': top['feature'],
                        'delta_llh_int': float(top['metric']),
                        'q_bh': float(top['q_bh']) if pd.notnull(top['q_bh']) else np.nan,
                        'sig_bh': bool(top['sig_bh']),
                        'n_participants': top['n_participants'],
                    })
    return pd.DataFrame(records)


def hedges_g_paired(x, y):
    """Small-sample corrected paired Hedges' g using participant-level differences."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd <= 0:
        return np.nan
    d = np.mean(diff) / sd
    n = len(diff)
    correction = 1.0 - (3.0 / (4.0 * n - 5.0))
    return float(correction * d)


def bootstrap_ci_hedges_g_paired(x, y, n_boot=5000, seed=42):
    """Percentile bootstrap CI for paired Hedges' g."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    local_rng = np.random.default_rng(seed)
    gs = []
    for _ in range(n_boot):
        idx = local_rng.integers(0, n, size=n)
        g = hedges_g_paired(x[idx], y[idx])
        if np.isfinite(g):
            gs.append(g)
    if not gs:
        return np.nan, np.nan
    return float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))


def summarize_group_target_mean(llh_delta_ctrl, group_label):
    """Mean Delta-LL over 14 feature tests (7 token + 7 segment) per target."""
    rows = []
    for label in ['log_TrtS', 'log_TrtT', 'log_Dur']:
        side = TARGET_TO_SIDE[label]
        vals = []
        for level_name in levels:
            df_tmp = llh_delta_ctrl[side][level_name][label]
            if df_tmp is not None and not df_tmp.empty:
                vals.extend(df_tmp['metric'].astype(float).tolist())
        if vals:
            rows.append({
                'group': group_label,
                'target': label,
                'mean_delta': float(np.mean(vals)),
            })
    return rows


def summarize_group_target_participant_mean(gp_results, group_label):
    """Participant-level mean Delta-LL over 14 feature tests per target."""
    rows = []
    for label in ['log_TrtS', 'log_TrtT', 'log_Dur']:
        side = TARGET_TO_SIDE[label]
        per_part_values = {}
        for level_name in levels:
            for gp, res_list in gp_results[side][level_name][label].items():
                if gp == 'control':
                    continue
                frames = []
                for res in res_list:
                    if res is None:
                        continue
                    df_delta = res.get('test_llh_delta_df')
                    if isinstance(df_delta, pd.DataFrame) and not df_delta.empty and 'translator_id' in df_delta.columns and 'delta' in df_delta.columns:
                        frames.append(df_delta[['translator_id', 'delta']].copy())
                if not frames:
                    continue
                joined = pd.concat(frames, ignore_index=True)
                ft_part_means = joined.groupby('translator_id', as_index=False)['delta'].mean()
                for _, rr in ft_part_means.iterrows():
                    pid = rr['translator_id']
                    per_part_values.setdefault(pid, []).append(float(rr['delta']))
        for pid, vals in per_part_values.items():
            if vals:
                rows.append({
                    'group': group_label,
                    'target': label,
                    'translator_id': pid,
                    'mean_delta': float(np.mean(vals)),
                })
    return rows


def build_stability_table(data, repeats=30, n_splits=3):
    """Repeated-CV stability summary by group and target."""
    print(f"Running repeated-CV stability analysis (repeats={repeats}, folds={n_splits})...")
    seed_list = [2026 + i for i in range(repeats)]
    series = {(g, t): [] for g in GROUP_LABELS for t in ['log_TrtS', 'log_TrtT', 'log_Dur']}
    participant_series = {}

    for i, seed in enumerate(seed_list, start=1):
        if i % 5 == 0 or i == 1 or i == repeats:
            print(f"  Stability repeat {i}/{repeats} (seed={seed})")
        for group_label in GROUP_LABELS:
            group_data = filter_group_data(data, group_label)
            gp_trains, gp_tests = get_train_test_split(group_data, n_splits=n_splits, random_seed=seed)
            if not gp_trains:
                continue
            gp_results = cross_validate(gp_trains, gp_tests, control_features, features)
            gp_llh_delta_ctrl = get_metric_vals(gp_results, 'test_llh_delta', concat_values=True)
            for row in summarize_group_target_mean(gp_llh_delta_ctrl, group_label):
                series[(group_label, row['target'])].append(row['mean_delta'])
            for row in summarize_group_target_participant_mean(gp_results, group_label):
                pkey = (row['group'], row['target'], row['translator_id'])
                participant_series.setdefault(pkey, []).append(row['mean_delta'])

    records = []
    for group_label in GROUP_LABELS:
        for target in ['log_TrtS', 'log_TrtT', 'log_Dur']:
            vals = np.asarray(series[(group_label, target)], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                records.append({
                    'group': group_label,
                    'target': target,
                    'mean_delta': np.nan,
                    'sd_delta': np.nan,
                    'pi_low': np.nan,
                    'pi_high': np.nan,
                    'pi_width': np.nan,
                    'sign_consistency': np.nan,
                    'n_repeats': 0,
                    'pi_crosses_zero': True,
                })
                continue
            pi_low = float(np.percentile(vals, 2.5))
            pi_high = float(np.percentile(vals, 97.5))
            sign_consistency = float(np.mean(np.sign(vals) == np.sign(np.mean(vals))))
            pi_width = float(pi_high - pi_low)
            records.append({
                'group': group_label,
                'target': target,
                'mean_delta': float(np.mean(vals)),
                'sd_delta': float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                'pi_low': pi_low,
                'pi_high': pi_high,
                'pi_width': pi_width,
                'sign_consistency': sign_consistency,
                'n_repeats': int(vals.size),
                'pi_crosses_zero': bool(pi_low <= 0 <= pi_high),
            })
    stability_df = pd.DataFrame(records)
    return stability_df, series, participant_series


def build_effect_size_table(participant_series, bootstrap_n=5000):
    """Paired participant-level Hedges' g + bootstrap CI for key contrasts."""
    contrasts = [
        ('all_translating', 'all_post-editing'),
        ('all_en-zh', 'all_zh-en'),
        ('post-editing_zh-en', 'all_post-editing'),
        ('post-editing_zh-en', 'post-editing_en-zh'),
    ]
    targets = ['log_TrtS', 'log_TrtT', 'log_Dur']
    records = []
    for g1, g2 in contrasts:
        for target in targets:
            g1_vals = {
                pid: float(np.mean(vals))
                for (grp, tgt, pid), vals in participant_series.items()
                if grp == g1 and tgt == target and len(vals) > 0
            }
            g2_vals = {
                pid: float(np.mean(vals))
                for (grp, tgt, pid), vals in participant_series.items()
                if grp == g2 and tgt == target and len(vals) > 0
            }
            common_ids = sorted(set(g1_vals.keys()).intersection(set(g2_vals.keys())))
            if common_ids:
                x = np.asarray([g1_vals[pid] for pid in common_ids], dtype=float)
                y = np.asarray([g2_vals[pid] for pid in common_ids], dtype=float)
            else:
                x = np.asarray([], dtype=float)
                y = np.asarray([], dtype=float)
            g = hedges_g_paired(x, y)
            ci_low, ci_high = bootstrap_ci_hedges_g_paired(x, y, n_boot=bootstrap_n, seed=42)
            records.append({
                'contrast': f'{g1} vs {g2}',
                'target': target,
                'effect_type': 'paired_participant_delta',
                'hedges_g_paired': g,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_pairs': int(len(x)),
                'unit': 'participant',
            })
    return pd.DataFrame(records)


def dump_figure_data(figures_dir, table_dict):
    """Write figure data tables to TSV + one combined TXT file."""
    if not table_dict:
        return

    txt_path = os.path.join(figures_dir, "all_figures_data.txt")
    with open(txt_path, "w", encoding="utf-8") as fw:
        fw.write("# All Figure Data (tab-separated sections)\n\n")
        for name, df in table_dict.items():
            safe_name = name.replace("/", "_")
            tsv_path = os.path.join(figures_dir, f"{safe_name}.tsv")
            if df is None or len(df) == 0:
                open(tsv_path, "w", encoding="utf-8").close()
                fw.write(f"## {name}\n")
                fw.write("(empty)\n\n")
                continue
            df.to_csv(tsv_path, sep="\t", index=False)
            fw.write(f"## {name}\n")
            fw.write(df.to_csv(sep="\t", index=False))
            fw.write("\n")

    print(f"  Saved {txt_path}")


def label_names(side, level, label, split=False):
    labels = {'log_TrtS': 'reading time', 'log_TrtT': 'reading time', 'log_Dur': 'duration'}
    side_names = {'src': 'source', 'tgt': 'target'}
    level_names = {'token': 'word'}
    if split:
        return f'{side} {level}\n{labels.get(label, label)}'
    else:
        elems = [side_names.get(side, side)]
        if level:
            elems.append(level_names.get(level, level))
        if label:
            elems.append(labels.get(label, label))
        return ' '.join(elems)


def group_display_name(group_label):
    return GROUP_DISPLAY_NAMES.get(group_label, str(group_label).replace('_', ' '))


def draw_significance_star(ax, x, y, color):
    ax.text(
        x, y, '*',
        color=color,
        fontsize=SIG_STAR_SIZE,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=35,
    )


def scatter_sig_marker(ax, x, y, color, marker, size, significant, zorder=20):
    facecolor = color if significant else 'none'
    ax.scatter(
        x,
        y,
        marker=marker,
        s=size,
        facecolors=facecolor,
        edgecolors=color,
        linewidths=1.2,
        zorder=zorder,
    )


def compute_shared_ylim(values, min_span=0.04):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-0.02, 0.02)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    span = hi - lo
    pad = max(0.0035, span * 0.15)
    ymin = lo - pad
    ymax = hi + pad
    if ymax - ymin < min_span:
        mid = (ymax + ymin) / 2
        ymin, ymax = mid - min_span / 2, mid + min_span / 2
    return (ymin, ymax)


def row_height_ratios(row_layout, separator_ratio=SPLIT_SEPARATOR_RATIO):
    return [separator_ratio if row_label is None else 1.0 for row_label in row_layout]


def grouped_row_ylim(row_label, row_ylim_map, out_base):
    if out_base == 'delta-llh-lang_pairs':
        return GROUP_MAIN_MANUAL_YLIM
    if out_base == 'delta-llh-lang_pairs-split':
        if row_label in GROUP_APPENDIX_SPECIAL_YLIMS:
            return GROUP_APPENDIX_SPECIAL_YLIMS[row_label]
        return row_ylim_map.get(row_label, (-0.02, 0.02))
    return row_ylim_map.get(row_label, (-0.02, 0.02))


def attention_row_ylim(row_label, row_ylim_map, out_base):
    if out_base == 'delta-llh-att-split' and row_label in ATT_APPENDIX_SPECIAL_YLIMS:
        return ATT_APPENDIX_SPECIAL_YLIMS[row_label]
    return row_ylim_map.get(row_label, (-0.02, 0.02))


def plot_main_delta_with_fdr(fig1_df):
    """Regenerate main Delta-LL panels with unified BH-FDR styling."""
    if fig1_df is None or fig1_df.empty:
        return

    target_specs = [
        ('src', 'log_TrtS', 'source reading time', 'delta-llh-src'),
        ('tgt', 'log_TrtT', 'target reading time', 'delta-llh-trtt'),
        ('tgt', 'log_Dur', 'target duration', 'delta-llh-dur'),
    ]
    shared_ylim = compute_shared_ylim(fig1_df['metric'].tolist())
    level_to_marker = {'token': MAINDELTA_TOKEN_MARKER, 'segment': MAINDELTA_SEG_MARKER}
    level_to_offset = {'token': -MAINDELTA_DODGE, 'segment': MAINDELTA_DODGE}

    for panel_idx, (side, label, title, stub) in enumerate(target_specs):
        fig, ax = plt.subplots(1, 1, figsize=MAINDELTA_PANEL_FIGSIZE)
        sub = fig1_df[(fig1_df['side'] == side) & (fig1_df['label'] == label)].copy()
        if sub.empty:
            plt.close(fig)
            continue
        sub['feature_id'] = sub['feature'].apply(lambda x: features[side].index(x) if x in features[side] else 999)
        sub = sub.sort_values(['feature_id', 'level'])
        fts = [ft for ft in features[side] if ft in set(sub['feature'])]
        x_positions = np.arange(len(fts))

        for level_name in ['token', 'segment']:
            level_df = sub[sub['level'] == level_name]
            for ft_idx, ft in enumerate(fts):
                ft_row = level_df[level_df['feature'] == ft]
                if ft_row.empty:
                    continue
                mean_y = float(ft_row['metric'].iloc[0])
                x_pos = x_positions[ft_idx] + level_to_offset[level_name]
                qval = float(ft_row['q_bh'].iloc[0]) if pd.notnull(ft_row['q_bh'].iloc[0]) else np.nan
                scatter_sig_marker(
                    ax,
                    x_pos,
                    mean_y,
                    MAINDELTA_POINT_COLOR,
                    level_to_marker[level_name],
                    MAINDELTA_POINT_SIZE,
                    bool(np.isfinite(qval) and qval < 0.05),
                )

        ax.set_xlim(-0.55, len(fts) - 0.45)
        ax.set_ylim(*shared_ylim)
        ax.axhline(0, ls='--', color=MAINDELTA_ZERO_LINE_COLOR, lw=MAINDELTA_ZERO_LINE_WIDTH)
        ax.grid(axis='y', color='0.94', linewidth=0.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([feature_names.get(ft, ft) for ft in fts],
                           size=MAINDELTA_XLABEL_SIZE, rotation=18, ha='right')
        if panel_idx == 0:
            ax.set_ylabel(r'$\Delta$ llh against control features', size=12)
            ax.tick_params(axis='y', labelsize=MAINDELTA_YTICK_LABELSIZE)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        ax.set_xlabel('')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('0.35')
            spine.set_linewidth(0.9)
        if panel_idx == 2:
            legend_elements = [
                Line2D([0], [0], marker=MAINDELTA_TOKEN_MARKER, color='w', label='token',
                       markerfacecolor=MAINDELTA_POINT_COLOR, markeredgecolor=MAINDELTA_POINT_COLOR,
                       linestyle='None', markersize=6),
                Line2D([0], [0], marker=MAINDELTA_SEG_MARKER, color='w', label='segment',
                       markerfacecolor=MAINDELTA_POINT_COLOR, markeredgecolor=MAINDELTA_POINT_COLOR,
                       linestyle='None', markersize=6),
                Line2D([0], [0], marker='o', color='w', label='significant',
                       markerfacecolor='black', markeredgecolor='black',
                       linestyle='None', markersize=6),
                Line2D([0], [0], marker='o', color='w', label='not significant',
                       markerfacecolor='white', markeredgecolor='black',
                       linestyle='None', markersize=6),
            ]
            ax.legend(
                handles=legend_elements,
                loc=MAINDELTA_LEGEND_LOC,
                fontsize=MAINDELTA_LEGEND_FONTSIZE,
                frameon=True,
                facecolor='white',
                edgecolor='0.85',
                borderpad=0.35,
                handletextpad=0.4,
            )
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/{stub}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{FIGURES_DIR}/{stub}.png', bbox_inches='tight', dpi=300)
        print(f"  Saved {FIGURES_DIR}/{stub}.png")
        plt.close(fig)


def plot_interaction_delta_from_table(fig_df, interaction_type, out_base):
    if fig_df is None or fig_df.empty:
        return
    sub_all = fig_df[fig_df['interaction_type'] == interaction_type].copy()
    if sub_all.empty:
        return

    target_specs = [
        ('src', 'log_TrtS', 'source reading time'),
        ('tgt', 'log_TrtT', 'target reading time'),
        ('tgt', 'log_Dur', 'target duration'),
    ]
    shared_ylim = compute_shared_ylim(sub_all['metric'].tolist(), min_span=0.02)
    level_to_marker = {'token': MAINDELTA_TOKEN_MARKER, 'segment': MAINDELTA_SEG_MARKER}
    level_to_offset = {'token': -MAINDELTA_DODGE, 'segment': MAINDELTA_DODGE}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True, constrained_layout=True)
    for panel_idx, (ax, (side, label, title)) in enumerate(zip(axes, target_specs)):
        sub = sub_all[(sub_all['side'] == side) & (sub_all['label'] == label)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub['feature_id'] = sub['feature'].apply(lambda x: features[side].index(x) if x in features[side] else 999)
        sub = sub.sort_values(['feature_id', 'level'])
        fts = [ft for ft in features[side] if ft in set(sub['feature'])]
        x_positions = np.arange(len(fts))
        for level_name in ['token', 'segment']:
            level_df = sub[sub['level'] == level_name]
            for ft_idx, ft in enumerate(fts):
                ft_row = level_df[level_df['feature'] == ft]
                if ft_row.empty:
                    continue
                mean_y = float(ft_row['metric'].iloc[0])
                sig = bool(ft_row['sig_bh'].iloc[0])
                scatter_sig_marker(
                    ax,
                    x_positions[ft_idx] + level_to_offset[level_name],
                    mean_y,
                    MAINDELTA_POINT_COLOR,
                    level_to_marker[level_name],
                    MAINDELTA_POINT_SIZE,
                    sig,
                )
        ax.set_xlim(-0.55, len(fts) - 0.45)
        ax.set_ylim(*shared_ylim)
        ax.axhline(0, ls='--', color=MAINDELTA_ZERO_LINE_COLOR, lw=MAINDELTA_ZERO_LINE_WIDTH)
        ax.grid(axis='y', color='0.94', linewidth=0.8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([feature_names.get(ft, ft) for ft in fts],
                           size=MAINDELTA_XLABEL_SIZE, rotation=18, ha='right')
        ax.set_title(title, size=MAINDELTA_TITLE_SIZE)
        if panel_idx == 0:
            ax.set_ylabel(r'Held-out interaction gain ($\Delta LL$)', size=12)
            ax.tick_params(axis='y', labelsize=MAINDELTA_YTICK_LABELSIZE)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('0.35')
            spine.set_linewidth(0.9)

    legend_elements = [
        Line2D([0], [0], marker=MAINDELTA_TOKEN_MARKER, color='w', label='token',
               markerfacecolor=MAINDELTA_POINT_COLOR, markeredgecolor=MAINDELTA_POINT_COLOR,
               linestyle='None', markersize=6),
        Line2D([0], [0], marker=MAINDELTA_SEG_MARKER, color='w', label='segment',
               markerfacecolor=MAINDELTA_POINT_COLOR, markeredgecolor=MAINDELTA_POINT_COLOR,
               linestyle='None', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='significant',
               markerfacecolor='black', markeredgecolor='black',
               linestyle='None', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='not significant',
               markerfacecolor='white', markeredgecolor='black',
               linestyle='None', markersize=6),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, INTERACTION_LEGEND_Y),
               fontsize=9, ncol=4, frameon=True)
    plt.savefig(f'{FIGURES_DIR}/{out_base}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{FIGURES_DIR}/{out_base}.png', bbox_inches='tight', dpi=300)
    print(f"  Saved {FIGURES_DIR}/{out_base}.png")
    plt.close(fig)


def should_render(selected_figs, figure_key):
    return selected_figs is None or figure_key in selected_figs


def load_saved_figure_tables():
    table_names = [
        'figure1_delta_llh_main',
        'figure2_coefficients',
        'figure3_delta_llh_att_grouped',
        'figure3_appendix_delta_llh_att_grouped_split',
        'figure4_att_shrinkage',
        'figure4_vif',
        'figure5_delta_llh_grouped',
        'figure5_appendix_delta_llh_grouped_split',
        'figure6_src_pos',
        'figure7_tgt_pos',
        'figure8_example_sentence',
        'figure_interaction_delta',
        'table_interaction_top',
        'table_bh_scope',
        'table_stability',
        'table_effect_sizes',
    ]
    tables = {}
    for name in table_names:
        path = os.path.join(FIGURES_DIR, f'{name}.tsv')
        tables[name] = pd.read_csv(path, sep='\t') if os.path.exists(path) else pd.DataFrame()
    return tables


def plot_coefficients_from_table(fig2_df):
    if fig2_df is None or fig2_df.empty:
        return

    predictor_color_map = {
        'length': '#4E79A7',
        'frequency': '#F28E2B',
        'position': '#59A14F',
        'predictor': '#B07AA1',
    }
    predictor_label_order = ['length', 'frequency', 'position', 'predictor']

    def predictor_kind(name):
        if name.endswith('_seg_len'):
            return 'length'
        if name.endswith('_freq'):
            return 'frequency'
        if name.endswith('_posq'):
            return 'position'
        return 'predictor'

    for label in ['log_TrtS', 'log_TrtT', 'log_Dur']:
        label_df = fig2_df[fig2_df['label'] == label].copy()
        if label_df.empty:
            continue
        side = label_df['side'].iloc[0]
        sig_list = [ft for ft in features[side] if ft in set(label_df['focus_feature'])]
        if not sig_list:
            continue

        plot_cols = len(sig_list)
        fig_w = max(5.8, 1.52 * plot_cols + 0.9)
        fig_h = 4.35
        fig, axes = plt.subplots(2, plot_cols, figsize=(fig_w, fig_h))
        if plot_cols == 1:
            axes = np.array(axes).reshape(2, 1)

        for row_id, level in enumerate(['token', 'segment']):
            level_df = label_df[label_df['level'] == level].copy()
            for ft_i, ft in enumerate(sig_list):
                ax = axes[row_id, ft_i]
                focus_df = level_df[level_df['focus_feature'] == ft].copy()
                if focus_df.empty:
                    ax.axis('off')
                    continue

                existing_fts = [f for f in control_features[side][level] + [ft]
                                if f in set(focus_df['predictor'])]
                focus_df['predictor_id'] = focus_df['predictor'].apply(
                    lambda x: existing_fts.index(x) if x in existing_fts else 999
                )
                focus_df = focus_df.sort_values('predictor_id')
                mean_coef = focus_df['mean_coef'].to_numpy(dtype=float)
                sem_coef = focus_df['ci95_halfwidth'].to_numpy(dtype=float)
                x = np.arange(len(existing_fts))

                bar_colors = [predictor_color_map[predictor_kind(pred_name)] for pred_name in existing_fts]
                ax.bar(x, mean_coef, color=bar_colors, width=0.72, edgecolor='none', zorder=2)
                ax.errorbar(
                    x, mean_coef, yerr=sem_coef, fmt='none',
                    ecolor='#2c333a', elinewidth=1.15, capsize=2.2, capthick=1.15, zorder=3
                )
                ax.set_xticks(x)
                ax.set_xticklabels([''] * len(x))
                ax.tick_params(axis='x', labelsize=9.5, length=3.0, width=0.9, pad=1.0)
                ax.tick_params(axis='y', labelsize=9.5, length=3.0, width=0.9, pad=1.0)
                ax.yaxis.tick_right()

                if row_id == 0:
                    ax.set_title(feature_names.get(ft, ft), size=15, pad=7)
                if ft_i == 0:
                    ax.set_ylabel('word' if level == 'token' else 'segment',
                                  labelpad=12, rotation=90, size=14)

                for spine_name in ['top', 'right', 'bottom', 'left']:
                    ax.spines[spine_name].set_visible(True)
                    ax.spines[spine_name].set_color('#bcc4ce')
                    ax.spines[spine_name].set_linewidth(1.0)
                ax.set_box_aspect(COEF_AX_BOX_ASPECT)

        legend_elements = [
            Patch(facecolor=predictor_color_map[k], edgecolor='none', label=k)
            for k in predictor_label_order
        ]
        fig.legend(
            handles=legend_elements, loc='upper center',
            bbox_to_anchor=(0.5, 0.99), fontsize=10.5, ncol=4,
            frameon=False, handlelength=1.8, columnspacing=1.1
        )
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.08, top=0.82, wspace=0.32, hspace=0.36)
        plt.savefig(f'{FIGURES_DIR}/coef_bar_{label}.pdf', dpi=300)
        plt.savefig(f'{FIGURES_DIR}/coef_bar_{label}.png', dpi=300)
        print(f"  Saved {FIGURES_DIR}/coef_bar_{label}.png")
        plt.close(fig)


def plot_attention_grouped_from_table(fig_df, row_layout, out_base):
    if fig_df is None or fig_df.empty:
        return

    n_rows_plot = len(row_layout)
    legend_y = ATT_APPENDIX_LEGEND_Y if out_base.endswith('-split') else ATT_LEGEND_Y
    sep_ratio = GROUP_SPLIT_SEPARATOR_RATIO
    hspace_val = GROUP_MAIN_HSPACE
    fig, axes = plt.subplots(
        n_rows_plot, 3,
        figsize=(14.2, max(4.8, 0.85 * n_rows_plot + 1.8)),
        sharex='col',
        gridspec_kw={
            'width_ratios': [1, 1, 1],
            'height_ratios': row_height_ratios(row_layout, separator_ratio=sep_ratio),
            'wspace': 0.10,
            'hspace': hspace_val,
        }
    )
    if n_rows_plot == 1:
        axes = axes.reshape(1, -1)

    baseline_colors = {'*ctrl': '#5f7fa6', '*ctrl+surp': '#d39a62'}
    level_markers = {'token': 'o', 'segment': '^'}
    combo_order = [('token', '*ctrl'), ('token', '*ctrl+surp'),
                   ('segment', '*ctrl'), ('segment', '*ctrl+surp')]
    offsets = np.linspace(-0.30, 0.30, len(combo_order))
    row_ylim = {
        group: compute_dynamic_ylim(fig_df[fig_df['group'] == group]['metric'].tolist())
        for group in fig_df['group'].dropna().unique()
    }

    for row_id, row_label in enumerate(row_layout):
        if row_label is None:
            for col_id in range(3):
                axes[row_id][col_id].axis('off')
            continue

        row_df = fig_df[fig_df['group'] == row_label].copy()
        for col_id, (side, label) in enumerate([(s, l) for s in sides for l in side2labels[s]]):
            ax = axes[row_id][col_id]
            sub = row_df[(row_df['side'] == side) & (row_df['label'] == label)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            fts_in_plot = [ft for ft in features[side] if ft in set(sub['feature'])]
            x_positions = np.arange(len(fts_in_plot))
            for combo_idx, (level_name, delta_name) in enumerate(combo_order):
                combo_df = sub[(sub['level'] == level_name) & (sub['delta'] == delta_name)]
                for ft_idx, ft in enumerate(fts_in_plot):
                    ft_df = combo_df[combo_df['feature'] == ft]
                    if ft_df.empty:
                        continue
                    mean_y = float(ft_df['metric'].iloc[0])
                    x_pos = x_positions[ft_idx] + offsets[combo_idx]
                    color = baseline_colors[delta_name]
                    qval = float(ft_df['q_bh'].iloc[0]) if pd.notnull(ft_df['q_bh'].iloc[0]) else np.nan
                    scatter_sig_marker(
                        ax,
                        x_pos,
                        mean_y,
                        color,
                        level_markers[level_name],
                        44,
                        bool(np.isfinite(qval) and qval < 0.05),
                    )

            ax.set_xlim(-0.55, len(fts_in_plot) - 0.45)
            ax.set_ylim(*attention_row_ylim(row_label, row_ylim, out_base))
            ax.axhline(0, ls='--', color=MAINDELTA_ZERO_LINE_COLOR, lw=MAINDELTA_ZERO_LINE_WIDTH)
            ax.grid(axis='y', color='0.94', linewidth=0.8)
            ax.set_ylabel('')
            ax.set_xlabel('')

            if col_id == 0:
                ax.tick_params(axis='y', labelsize=ATT_YTICK_LABELSIZE)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if col_id == 2:
                ax.set_ylabel(group_display_name(row_label), rotation=0, fontsize=11,
                              ha='left', va='center', labelpad=28)
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_label_coords(ATT_ROW_LABEL_RIGHT_X, 0.5)

            if row_id == 0:
                ax.set_title(label_names(side, None, label), size=13)

            if row_id == n_rows_plot - 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([feature_names.get(ft, ft) for ft in fts_in_plot],
                                   size=12, rotation=22, ha='right')
            else:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([])

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Token',
               markerfacecolor='black', markeredgecolor='black', markersize=7),
        Line2D([0], [0], marker='^', color='w', label='Segment',
               markerfacecolor='black', markeredgecolor='black', markersize=7),
        Line2D([0], [0], marker='o', color=baseline_colors['*ctrl'], label='Control only',
               markerfacecolor=baseline_colors['*ctrl'], markeredgecolor=baseline_colors['*ctrl'],
               linestyle='None', markersize=7),
        Line2D([0], [0], marker='o', color=baseline_colors['*ctrl+surp'], label='Control + surprisal',
               markerfacecolor=baseline_colors['*ctrl+surp'], markeredgecolor=baseline_colors['*ctrl+surp'],
               linestyle='None', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Significant',
               markerfacecolor='black', markeredgecolor='black', linestyle='None', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Not significant',
               markerfacecolor='white', markeredgecolor='black', linestyle='None', markersize=7),
    ]
    fig.text(0.02, 0.5, r'$\Delta$ llh per word/segment', va='center', rotation='vertical', size=12)
    lgd = fig.legend(handles=legend_elements, loc='lower center',
                     bbox_to_anchor=(0.5, legend_y), fontsize=11, ncol=6, frameon=True)
    fig.tight_layout(rect=[0.08, 0.08, 1.0, 1.0])
    plt.savefig(f'{FIGURES_DIR}/{out_base}.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    plt.savefig(f'{FIGURES_DIR}/{out_base}.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    print(f"  Saved {FIGURES_DIR}/{out_base}.png")
    plt.close(fig)


def plot_att_shrinkage_from_table(shrink_df):
    if shrink_df is None or shrink_df.empty:
        return

    row_order = [
        'Source reading (token)',
        'Source reading (segment)',
        'Target reading (token)',
        'Target reading (segment)',
        'Target duration (token)',
        'Target duration (segment)',
    ]
    y_values = np.arange(len(row_order), dtype=float) * SHRINK_Y_STEP
    y_map = {name: y for name, y in zip(row_order, y_values)}

    lm_df = shrink_df[shrink_df['baseline_model'] == 'LM'].copy()
    nmt_df = shrink_df[shrink_df['baseline_model'] == 'NMT'].copy()

    fig, ax = plt.subplots(1, 1, figsize=SHRINK_FIGSIZE, constrained_layout=True)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    rng_cloud = np.random.default_rng(42)

    def parse_vals(raw):
        if raw is None:
            return []
        if isinstance(raw, float) and np.isnan(raw):
            return []
        if isinstance(raw, (list, tuple, np.ndarray)):
            return [float(v) for v in raw if np.isfinite(v)]
        if isinstance(raw, str):
            txt = raw.strip()
            if not txt:
                return []
            try:
                vals = json.loads(txt)
            except json.JSONDecodeError:
                return []
            if isinstance(vals, (list, tuple)):
                return [float(v) for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        return []

    def draw_half_cloud(vals, y_center, color, half):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            return
        if float(np.nanstd(arr)) < 1e-12:
            return
        vp = ax.violinplot(
            [arr],
            positions=[y_center],
            vert=False,
            widths=SHRINK_CLOUD_WIDTH,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = vp['bodies'][0]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(0.9)
        body.set_zorder(1.0)
        verts = body.get_paths()[0].vertices
        if half == 'upper':
            verts[:, 1] = np.minimum(verts[:, 1], y_center)
        else:
            verts[:, 1] = np.maximum(verts[:, 1], y_center)

    def draw_side(df, sign):
        row_lookup = {row['cell']: row for _, row in df.iterrows()}
        for cell in row_order:
            y = y_map[cell]
            if cell not in row_lookup:
                if sign > 0 and cell.startswith('Source'):
                    ax.text(
                        x_lim * 0.10, y, 'n/a',
                        ha='left', va='center', fontsize=8.5,
                        color='#000000', style='italic', zorder=5
                    )
                continue

            row = row_lookup[cell]
            ctrl = float(row['control_only'])
            surp = float(row['control_plus_surprisal'])
            delta = ctrl - surp
            x_ctrl = sign * ctrl
            x_surp = sign * surp

            ax.plot(
                [x_surp, x_ctrl], [y, y],
                color=SHRINK_LINE_COLOR, lw=SHRINK_LINE_WIDTH, alpha=1.0,
                solid_capstyle='round', zorder=2
            )
            ctrl_vals = parse_vals(row.get('control_values_json'))
            surp_vals = parse_vals(row.get('surprisal_values_json'))
            ctrl_arr = sign * np.asarray(ctrl_vals, dtype=float) if ctrl_vals else np.array([], dtype=float)
            surp_arr = sign * np.asarray(surp_vals, dtype=float) if surp_vals else np.array([], dtype=float)
            if ctrl_vals:
                draw_half_cloud(ctrl_arr, y - SHRINK_CLOUD_Y_OFFSET, SHRINK_CTRL_COLOR, half='upper')
            if surp_vals:
                draw_half_cloud(surp_arr, y + SHRINK_CLOUD_Y_OFFSET, SHRINK_SURP_COLOR, half='lower')
            if len(ctrl_arr) > 0:
                y_jit = y + rng_cloud.normal(-SHRINK_RAIN_Y_OFFSET, SHRINK_RAIN_Y_JITTER, size=len(ctrl_arr))
                ax.scatter(
                    ctrl_arr, y_jit,
                    color=SHRINK_CTRL_COLOR, s=SHRINK_RAIN_POINT_SIZE, alpha=SHRINK_RAIN_POINT_ALPHA,
                    edgecolors='white', linewidths=0.25, zorder=1.6
                )
            if len(surp_arr) > 0:
                y_jit = y + rng_cloud.normal(SHRINK_RAIN_Y_OFFSET, SHRINK_RAIN_Y_JITTER, size=len(surp_arr))
                ax.scatter(
                    surp_arr, y_jit,
                    color=SHRINK_SURP_COLOR, s=SHRINK_RAIN_POINT_SIZE, alpha=SHRINK_RAIN_POINT_ALPHA,
                    edgecolors='white', linewidths=0.25, zorder=1.6
                )
            ax.scatter(
                x_ctrl, y, color=SHRINK_CTRL_MEAN_COLOR, s=SHRINK_POINT_SIZE,
                edgecolors=SHRINK_CTRL_MEAN_EDGE_COLOR, linewidths=SHRINK_MEAN_EDGE_WIDTH,
                zorder=4, marker='o'
            )
            ax.scatter(
                x_surp, y, color=SHRINK_SURP_MEAN_COLOR, s=SHRINK_POINT_SIZE,
                    edgecolors=SHRINK_SURP_MEAN_EDGE_COLOR, linewidths=SHRINK_MEAN_EDGE_WIDTH,
                    zorder=4, marker='D'
                )

            text_x = sign * (max(ctrl, surp) + x_lim * 0.03)
            text_x = float(np.clip(text_x, -x_lim * 0.97, x_lim * 0.97))
            ax.text(
                text_x, y, f'\u0394{delta:.5f}',
                ha='right' if sign < 0 else 'left',
                va='center', fontsize=SHRINK_DELTA_SIZE, color=SHRINK_DELTA_COLOR, zorder=5
            )

    all_vals = pd.concat(
        [
            lm_df[['control_only', 'control_plus_surprisal']],
            nmt_df[['control_only', 'control_plus_surprisal']],
        ],
        axis=0,
        ignore_index=True,
    ).to_numpy(dtype=float)
    cloud_vals = []
    for _, row in shrink_df.iterrows():
        cloud_vals.extend(parse_vals(row.get('control_values_json')))
        cloud_vals.extend(parse_vals(row.get('surprisal_values_json')))
    if cloud_vals:
        cloud_max = float(np.nanmax(np.abs(np.array(cloud_vals, dtype=float))))
        mean_max = float(np.nanmax(np.abs(all_vals))) if all_vals.size else 0.01
        max_abs = max(cloud_max, mean_max, 0.01)
    else:
        max_abs = np.nanmax(np.abs(all_vals)) if all_vals.size else 0.01
    x_lim = max_abs * 1.16

    ax.set_xlim(-x_lim, x_lim)

    draw_side(lm_df, sign=-1.0)
    draw_side(nmt_df, sign=1.0)

    ticks = np.linspace(-x_lim, x_lim, 7)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{abs(t):.3f}' for t in ticks], fontsize=10)
    ax.tick_params(axis='x', labelsize=10, colors='#000000')
    ax.set_xlabel(r'Mean $\Delta$ LL', fontsize=11, color='#000000')

    y_pos = y_values
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_order, fontsize=SHRINK_LABELSIZE)
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0, colors='#000000')

    ax.axvline(0.0, color=SHRINK_ZERO_LINE_COLOR, lw=0.7, ls='--', zorder=2.2)
    ax.grid(axis='x', color=SHRINK_GRID_COLOR, linewidth=0.75, linestyle=':', alpha=1.0)
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_color(SHRINK_SPINE_COLOR)
        ax.spines[side].set_linewidth(1.0)

    ax.text(0.23, 1.02, 'LM baseline', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=SHRINK_TITLE_SIZE,
            color=SHRINK_TITLE_COLOR, fontweight='semibold')
    ax.text(0.77, 1.02, 'NMT baseline', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=SHRINK_TITLE_SIZE,
            color=SHRINK_TITLE_COLOR, fontweight='semibold')

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=7.5,
               markerfacecolor=SHRINK_CTRL_MEAN_COLOR, markeredgecolor=SHRINK_CTRL_MEAN_EDGE_COLOR,
               markeredgewidth=SHRINK_MEAN_EDGE_WIDTH, label='Control only'),
        Line2D([0], [0], marker='D', linestyle='None', markersize=7.5,
               markerfacecolor=SHRINK_SURP_MEAN_COLOR, markeredgecolor=SHRINK_SURP_MEAN_EDGE_COLOR,
               markeredgewidth=SHRINK_MEAN_EDGE_WIDTH, label='Control + surprisal'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              ncol=2, frameon=True, facecolor='white', edgecolor='#d0d5dc',
              framealpha=1.0, fontsize=9)

    plt.savefig(f'{FIGURES_DIR}/att-shrinkage.pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(f'{FIGURES_DIR}/att-shrinkage.png', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"  Saved {FIGURES_DIR}/att-shrinkage.png")
    plt.close(fig)


def plot_vif_from_table(vif_df):
    if vif_df is None or vif_df.empty:
        return

    fig, axes_vif = plt.subplots(2, 2, figsize=(8.2, 5.6), gridspec_kw={'wspace': 0.28, 'hspace': 0.55})
    panel_file_stub = {
        ('src', 'token'): 'vif_src_token',
        ('src', 'segment'): 'vif_src_segment',
        ('tgt', 'token'): 'vif_tgt_token',
        ('tgt', 'segment'): 'vif_tgt_segment',
    }
    for rid, side in enumerate(sides):
        for cid, level in enumerate(['token', 'segment']):
            ax = axes_vif[rid, cid]
            sub = vif_df[(vif_df['side'] == side) & (vif_df['level'] == level)].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            fts_list = [ft for ft in features[side] if ft in set(sub['feature'])]
            corr_rows = []
            for row_name in ['length', 'frequency', 'position', 'predictor']:
                row_vals = []
                for ft in fts_list:
                    ft_row = sub[(sub['feature'] == ft) & (sub['vif_row'] == row_name)]
                    row_vals.append(float(ft_row['vif_value'].iloc[0]) if not ft_row.empty else np.nan)
                corr_rows.append(row_vals)
            corr_mtx = np.array(corr_rows)
            ft_labels = [feature_names.get(ft, ft) for ft in fts_list]

            sns.heatmap(
                corr_mtx,
                ax=ax,
                annot=True,
                annot_kws={"fontsize": 10},
                cbar=None,
                cmap='Blues',
                vmax=2.3,
                vmin=1,
            )
            ax.set_ylabel('')
            ax.set_yticklabels(['length', 'frequency', 'position', 'predictor'], rotation=0)
            ax.set_xticklabels(ft_labels, rotation=0, size=12)
            ax.xaxis.tick_top()

            fig_panel, ax_panel = plt.subplots(1, 1, figsize=(4.0, 2.6), constrained_layout=True)
            sns.heatmap(
                corr_mtx,
                ax=ax_panel,
                annot=True,
                annot_kws={"fontsize": 10},
                cbar=None,
                cmap='Blues',
                vmax=2.3,
                vmin=1,
            )
            ax_panel.set_ylabel('')
            ax_panel.set_yticklabels(['length', 'frequency', 'position', 'predictor'], rotation=0)
            ax_panel.set_xticklabels(ft_labels, rotation=0, size=12)
            ax_panel.xaxis.tick_top()
            stub = panel_file_stub[(side, level)]
            plt.savefig(f'{FIGURES_DIR}/{stub}.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(f'{FIGURES_DIR}/{stub}.png', bbox_inches='tight', dpi=300)
            print(f"  Saved {FIGURES_DIR}/{stub}.png")
            plt.close(fig_panel)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/vif.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{FIGURES_DIR}/vif.png', bbox_inches='tight', dpi=300)
    print(f"  Saved {FIGURES_DIR}/vif.png")
    plt.close(fig)


def plot_grouped_delta_from_table(fig_df, row_layout, out_base):
    if fig_df is None or fig_df.empty:
        return

    n_rows_plot = len(row_layout)
    legend_y = GROUP_APPENDIX_LEGEND_Y if out_base.endswith('-split') else GROUP_LEGEND_Y
    fig, axes_lp = plt.subplots(
        n_rows_plot, 3,
        figsize=(14.0, max(4.8, 0.85 * n_rows_plot + 1.8)),
        sharex='col',
        gridspec_kw={
            'width_ratios': [1, 1, 1],
            'height_ratios': row_height_ratios(row_layout, separator_ratio=GROUP_SPLIT_SEPARATOR_RATIO),
            'wspace': 0.08,
            'hspace': GROUP_MAIN_HSPACE,
        }
    )
    if n_rows_plot == 1:
        axes_lp = axes_lp.reshape(1, -1)

    offsets = {'token': -0.10, 'segment': 0.10}
    markers = {'token': 'o', 'segment': '^'}
    row_ylim = {
        group: compute_dynamic_ylim(fig_df[fig_df['group'] == group]['metric'].tolist())
        for group in fig_df['group'].dropna().unique()
    }

    for row_id, row_label in enumerate(row_layout):
        if row_label is None:
            for col_id in range(3):
                axes_lp[row_id][col_id].axis('off')
            continue

        row_df = fig_df[fig_df['group'] == row_label].copy()
        for col_id, (side, label) in enumerate([(s, l) for s in sides for l in side2labels[s]]):
            ax = axes_lp[row_id][col_id]
            sub = row_df[(row_df['side'] == side) & (row_df['label'] == label)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            fts = [ft for ft in features[side] if ft in set(sub['feature'])]
            x_positions = np.arange(len(fts))
            for level_name in ['token', 'segment']:
                combo_df = sub[sub['level'] == level_name]
                for ft_idx, ft in enumerate(fts):
                    ft_df = combo_df[combo_df['feature'] == ft]
                    if ft_df.empty:
                        continue
                    mean_y = float(ft_df['metric'].iloc[0])
                    x_pos = x_positions[ft_idx] + offsets[level_name]
                    qval = float(ft_df['q_bh'].iloc[0]) if pd.notnull(ft_df['q_bh'].iloc[0]) else np.nan
                    scatter_sig_marker(
                        ax,
                        x_pos,
                        mean_y,
                        '#4a4a4a',
                        markers[level_name],
                        42,
                        bool(np.isfinite(qval) and qval < 0.05),
                    )

            ax.set_xlim(-0.55, len(fts) - 0.45)
            ax.set_ylim(*grouped_row_ylim(row_label, row_ylim, out_base))
            ax.axhline(0, ls='--', color=MAINDELTA_ZERO_LINE_COLOR, lw=MAINDELTA_ZERO_LINE_WIDTH)
            ax.grid(axis='y', color='0.94', linewidth=0.8)
            ax.set_ylabel('')
            ax.set_xlabel('')

            if col_id == 0:
                ax.tick_params(axis='y', labelsize=GROUP_YTICK_LABELSIZE)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if col_id == 2:
                ax.set_ylabel(group_display_name(row_label), rotation=0, fontsize=11,
                              ha='left', va='center', labelpad=28)
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_label_coords(GROUP_ROW_LABEL_RIGHT_X, 0.5)

            if row_id == 0:
                ax.set_title(label_names(side, None, label, split=False), size=13)

            if row_id == n_rows_plot - 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([feature_names.get(ft, ft) for ft in fts], size=12, rotation=22, ha='right')
            else:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([])

    txt = fig.text(0.08, 0.5, r'$\Delta$ llh against control features', va='center', rotation='vertical', size=12)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Token',
               markerfacecolor='black', markeredgecolor='black', markersize=7),
        Line2D([0], [0], marker='^', color='w', label='Segment',
               markerfacecolor='black', markeredgecolor='black', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Significant',
               markerfacecolor='black', markeredgecolor='black', linestyle='None', markersize=7),
        Line2D([0], [0], marker='o', color='w', label='Not significant',
               markerfacecolor='white', markeredgecolor='black', linestyle='None', markersize=7),
    ]
    lgd = fig.legend(handles=legend_elements, loc='lower center',
                     bbox_to_anchor=(0.5, legend_y), fontsize=11, ncol=4, frameon=True)
    fig.tight_layout(rect=[0.08, 0.08, 1.0, 1.0])
    plt.savefig(f'{FIGURES_DIR}/{out_base}.pdf',
                bbox_extra_artists=(lgd, txt), bbox_inches='tight', dpi=300)
    plt.savefig(f'{FIGURES_DIR}/{out_base}.png',
                bbox_extra_artists=(lgd, txt), bbox_inches='tight', dpi=300)
    print(f"  Saved {FIGURES_DIR}/{out_base}.png")
    plt.close(fig)


def _reconstruct_longitudinal_feature_frame(pos_df):
    if pos_df is None or pos_df.empty:
        return pd.DataFrame()
    temp = pos_df.copy()
    temp['obs_id'] = temp.groupby(['feature', 'pos']).cumcount()
    wide = temp.pivot(index=['pos', 'obs_id'], columns='feature', values='value').reset_index()
    wide.columns.name = None
    return wide


def plot_src_pos_from_table(pos_df):
    if pos_df is None or pos_df.empty:
        return

    wide = _reconstruct_longitudinal_feature_frame(pos_df)
    if wide.empty or 'log_TrtS' not in wide.columns:
        return
    valid_pos = wide['pos'].value_counts()[lambda s: s >= 20].index
    wide = wide[wide['pos'].isin(valid_pos)].copy()
    order = list(wide[['pos', 'log_TrtS']].groupby('pos').mean().sort_values('log_TrtS').index)
    src_sig_fts = [ft for ft in features['src'] if ft in set(pos_df['feature']) and ft != 'log_TrtS']
    fts = ['log_TrtS'] + src_sig_fts

    fig, axes_pos = plt.subplots(1, len(fts), figsize=(min(9, len(fts) * 2.5), 3.5),
                                 sharey=True, sharex=False, constrained_layout=True,
                                 gridspec_kw={'wspace': 0.0})
    if len(fts) == 1:
        axes_pos = [axes_pos]

    for i, (ax, ft) in enumerate(zip(axes_pos, fts)):
        if ft not in wide.columns:
            continue
        new_df = pd.DataFrame({'pos': wide['pos'], ft: wide[ft].values}).dropna()
        new_df['rank'] = new_df['pos'].apply(lambda x: order.index(x) if x in order else 999)
        new_df = new_df.sort_values(by='rank')
        panel_color = POS_PRIMARY_COLOR if i == 0 else POS_SECONDARY_COLOR
        sns.pointplot(data=new_df, x=ft, y='pos', ax=ax, capsize=.0, join=False, color=panel_color, scale=0.9)
        for line in ax.lines:
            line.set_color(POS_ERR_COLOR)
            line.set_linewidth(1.0)
        ax.tick_params(axis='y', which='major', labelsize=POS_TICK_SIZE)
        if i > 0:
            ax.get_yaxis().set_visible(False)
            ax.set_xlabel(feature_names.get(ft, 'src reading time'), size=POS_LABEL_SIZE)
            corr_df = wide[['log_TrtS', ft]].dropna()
            pcorr = pearsonr(corr_df[ft], corr_df['log_TrtS']) if len(corr_df) > 1 else None
            pstr = (r'$\rho=$' + f'${pcorr.statistic:.2f}^*$') if pcorr and pcorr.pvalue < .001 else (
                r'$\rho=$' + f'${pcorr.statistic:.2f}$' if pcorr else ''
            )
            ax.set_title(pstr, size=POS_TITLE_SIZE, color=POS_PRIMARY_COLOR)
        else:
            ax.set_ylabel('')
            ax.set_xlabel(feature_names.get(ft, 'src reading\ntime'), size=POS_LABEL_SIZE)
        for spine in ax.spines.values():
            spine.set_color(POS_SPINE_COLOR)
            spine.set_linewidth(POS_SPINE_WIDTH)

    plt.savefig(f'{FIGURES_DIR}/src-POS.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{FIGURES_DIR}/src-POS.png', bbox_inches='tight', dpi=300)
    print(f"  Saved {FIGURES_DIR}/src-POS.png")
    plt.close(fig)


def plot_tgt_pos_from_table(pos_df):
    if pos_df is None or pos_df.empty:
        return

    wide = _reconstruct_longitudinal_feature_frame(pos_df)
    if wide.empty or 'log_Dur' not in wide.columns:
        return
    valid_pos = wide['pos'].value_counts()[lambda s: s >= 20].index
    wide = wide[wide['pos'].isin(valid_pos)].copy()
    order = list(wide[['pos', 'log_Dur']].groupby('pos').mean().sort_values('log_Dur').index)
    fts = ['log_Dur', 'xent'] + [ft for ft in features['tgt'] if ft in set(pos_df['feature']) and ft not in ['log_Dur', 'xent', 'tgt_surp']]
    fts = [ft for ft in fts if ft in set(pos_df['feature'])]

    fig, axes_tpos = plt.subplots(1, len(fts), figsize=(min(15, len(fts) * 2.5), 3.5),
                                  sharex=False, sharey=True, constrained_layout=True,
                                  gridspec_kw={'wspace': 0.1})
    if len(fts) == 1:
        axes_tpos = [axes_tpos]

    for row_idx, ft in enumerate(fts):
        ax = axes_tpos[row_idx]
        new_df = pd.DataFrame({'pos': wide['pos'], ft: wide[ft].values}).dropna()
        new_df['rank'] = new_df['pos'].apply(lambda x: order.index(x) if x in order else 999)
        new_df = new_df.sort_values(by='rank')
        panel_color = POS_PRIMARY_COLOR if row_idx == 0 else POS_SECONDARY_COLOR
        sns.pointplot(data=new_df, x=ft, y='pos', ax=ax, capsize=.0, join=False, color=panel_color, scale=0.9)
        for line in ax.lines:
            line.set_color(POS_ERR_COLOR)
            line.set_linewidth(1.0)
        ax.tick_params(axis='y', which='major', labelsize=POS_TICK_SIZE)
        if row_idx > 0:
            ax.get_yaxis().set_visible(False)
            ax.set_xlabel(feature_names.get(ft, 'tgt duration'), size=POS_LABEL_SIZE)
            corr_df = wide[['log_Dur', ft]].dropna()
            pcorr = pearsonr(corr_df[ft], corr_df['log_Dur']) if len(corr_df) > 1 else None
            pstr = (r'$\rho=$' + f'${pcorr.statistic:.2f}^*$') if pcorr and pcorr.pvalue < .001 else (
                r'$\rho=$' + f'${pcorr.statistic:.2f}$' if pcorr else ''
            )
            ax.set_title(pstr, size=POS_TITLE_SIZE, color=POS_PRIMARY_COLOR)
        else:
            ax.set_ylabel('')
            ax.set_xlabel(feature_names.get(ft, 'tgt duration'), size=POS_LABEL_SIZE)
        for spine in ax.spines.values():
            spine.set_color(POS_SPINE_COLOR)
            spine.set_linewidth(POS_SPINE_WIDTH)
        ax.set_box_aspect(TGT_POS_PANEL_BOX_ASPECT)

    plt.savefig(f'{FIGURES_DIR}/tgt-POS.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{FIGURES_DIR}/tgt-POS.png', bbox_inches='tight', dpi=300)
    print(f"  Saved {FIGURES_DIR}/tgt-POS.png")
    plt.close(fig)


def plot_example_sentence_from_table(df_ex):
    if df_ex is None or df_ex.empty:
        return

    fts_ex = ['log_Dur', 'xent', 'y2srcseq_ent']
    ft_name_ex = {'log_Dur': 'translation\nduration', 'xent': 'translation\nsurprisal',
                  'y2srcseq_ent': 'cross-attention\nentropy'}
    fig, axes_ex = plt.subplots(1, len(fts_ex), figsize=(min(15, len(fts_ex) * 2.5), 3.5), sharey=True,
                                constrained_layout=True, gridspec_kw={'wspace': 0.1})
    for i, ft in enumerate(fts_ex):
        ax = axes_ex[i]
        if ft not in df_ex.columns:
            continue
        new_df = pd.DataFrame({'src_tokens': df_ex.src_tokens, 'values': df_ex[ft].values})
        sns.pointplot(data=new_df, y='src_tokens', x='values', color=POS_ERR_COLOR,
                      dodge=True, join=False, ax=ax, scale=0.9)
        sns.pointplot(data=new_df, y='src_tokens', x='values', color=POS_PRIMARY_COLOR,
                      ax=ax, errorbar=None, join=False, scale=0.9)
        plt.setp(ax.collections, zorder=100, label="")
        ax.set_ylabel('')
        ax.tick_params(axis='y', which='major', labelsize=POS_TICK_SIZE)
        ax.tick_params(axis='x', which='major', labelsize=POS_TICK_SIZE)
        if i > 0:
            ax.get_yaxis().set_visible(False)
            corr_df = df_ex[['log_Dur', ft]].dropna()
            pcorr = pearsonr(corr_df[ft], corr_df['log_Dur']) if len(corr_df) > 1 else None
            pstr = (r'$\rho=$' + f'${pcorr.statistic:.2f}^*$') if pcorr and pcorr.pvalue < .05 else (
                r'$\rho=$' + f'${pcorr.statistic:.2f}$' if pcorr else ''
            )
            ax.set_title(pstr, size=POS_TITLE_SIZE, color=POS_PRIMARY_COLOR)
        ax.set_xlabel(ft_name_ex.get(ft, ft), size=POS_LABEL_SIZE)
        for spine in ax.spines.values():
            spine.set_color(POS_SPINE_COLOR)
            spine.set_linewidth(POS_SPINE_WIDTH)
        ax.set_box_aspect(EXAMPLE_PANEL_BOX_ASPECT)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/tgt-example.pdf', bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.savefig(f'{FIGURES_DIR}/tgt-example.png', bbox_inches='tight', dpi=300, pad_inches=0.05)
    print(f"  Saved {FIGURES_DIR}/tgt-example.png")
    plt.close(fig)


def render_figures_from_saved_tables(selected_figs=None):
    figure_tables = load_saved_figure_tables()
    if should_render(selected_figs, 'main_delta'):
        plot_main_delta_with_fdr(figure_tables['figure1_delta_llh_main'])
    if should_render(selected_figs, 'coef_bars'):
        plot_coefficients_from_table(figure_tables['figure2_coefficients'])
    if should_render(selected_figs, 'att_delta'):
        plot_attention_grouped_from_table(
            figure_tables['figure3_delta_llh_att_grouped'], MAIN_ROW_LAYOUT, 'delta-llh-att'
        )
    if should_render(selected_figs, 'att_delta_appendix'):
        plot_attention_grouped_from_table(
            figure_tables['figure3_appendix_delta_llh_att_grouped_split'],
            APPENDIX_ROW_LAYOUT,
            'delta-llh-att-split',
        )
    if should_render(selected_figs, 'att_shrinkage'):
        plot_att_shrinkage_from_table(figure_tables['figure4_att_shrinkage'])
    if should_render(selected_figs, 'vif'):
        plot_vif_from_table(figure_tables['figure4_vif'])
    if should_render(selected_figs, 'group_delta'):
        plot_grouped_delta_from_table(
            figure_tables['figure5_delta_llh_grouped'], MAIN_ROW_LAYOUT, 'delta-llh-lang_pairs'
        )
    if should_render(selected_figs, 'group_delta_appendix'):
        plot_grouped_delta_from_table(
            figure_tables['figure5_appendix_delta_llh_grouped_split'],
            APPENDIX_ROW_LAYOUT,
            'delta-llh-lang_pairs-split',
        )
    if should_render(selected_figs, 'interaction_task'):
        plot_interaction_delta_from_table(
            figure_tables['figure_interaction_delta'], 'task', 'interaction-task'
        )
    if should_render(selected_figs, 'interaction_direction'):
        plot_interaction_delta_from_table(
            figure_tables['figure_interaction_delta'], 'direction', 'interaction-direction'
        )
    if should_render(selected_figs, 'src_pos'):
        plot_src_pos_from_table(figure_tables['figure6_src_pos'])
    if should_render(selected_figs, 'tgt_pos'):
        plot_tgt_pos_from_table(figure_tables['figure7_tgt_pos'])
    if should_render(selected_figs, 'example_sentence'):
        plot_example_sentence_from_table(figure_tables['figure8_example_sentence'])


# ─── Main ────────────────────────────────────────────────────────

def main(render_after=True):
    print("Loading data...")
    figure_tables = {}
    data = get_data(DATA_DIR, standardize=True)
    preprocessing_summary = build_preprocessing_summary(data)
    figure_tables['table_preprocessing_summary'] = preprocessing_summary
    for side in sides:
        for level in levels:
            print(f"  {side}-{level}: {len(data[side][level])} rows")
    if not preprocessing_summary.empty:
        print("Preprocessing summary:")
        for _, row in preprocessing_summary.iterrows():
            print(
                f"  {row['side']}-{row['level']} {row['label']}: "
                f"{int(row['valid_rows'])}/{int(row['total_rows'])} valid "
                f"({row['valid_rate']:.3f})"
            )

    print("Creating train/test splits...")
    trains, tests = get_train_test_split(data, n_splits=MAIN_N_SPLITS)

    print("Running cross-validation with control features...")
    all_results = cross_validate(trains, tests, control_features, features)

    print("Running significance tests...")
    sigfig = get_delta_sigfig(all_results)

    # Determine significant features
    significant_features = {side: {} for side in sides}
    for side in sides:
        for label in side2labels[side]:
            token_sig = set(ft for ft, sig in sigfig[side]['token'][label].items() if sig.pvalue < .05)
            seg_sig = set(ft for ft, sig in sigfig[side]['segment'][label].items() if sig.pvalue < .05)
            sig_fts = list(sorted(token_sig.union(seg_sig),
                                  key=lambda x: features[side].index(x) if x in features[side] else 999))
            significant_features[side][label] = sig_fts

    print("Significant features:", significant_features)

    # ─── PLOT 1: Delta log-likelihood pointplot (Cell 18 style) ───
    print("Generating delta-llh plot (main figure)...")
    llh_delta = get_metric_vals(all_results, 'test_llh_delta', concat_values=True)
    fig1_records = []
    for side in sides:
        for label in side2labels[side]:
            for level in ['token', 'segment']:
                df_tmp = llh_delta[side][level][label].copy()
                if df_tmp.empty:
                    continue
                df_tmp['side'] = side
                df_tmp['label'] = label
                df_tmp['level'] = level
                df_tmp['feature'] = df_tmp['predictor'].apply(lambda x: x.split('+')[-1])
                df_tmp['p_raw'] = df_tmp['feature'].apply(
                    lambda ft: sigfig[side][level][label][ft].pvalue
                    if ft in sigfig[side][level][label] else np.nan)
                df_tmp['n_participants'] = df_tmp['feature'].apply(
                    lambda ft: sigfig[side][level][label][ft].n_participants
                    if ft in sigfig[side][level][label] else np.nan)
                df_tmp['q_bh'] = np.nan
                df_tmp['sig_bh'] = False
                fig1_records.append(df_tmp[['side', 'label', 'level', 'predictor', 'feature', 'metric', 'p_raw', 'n_participants', 'q_bh', 'sig_bh']])
    figure_tables['figure1_delta_llh_main'] = pd.concat(fig1_records, ignore_index=True) if fig1_records else pd.DataFrame()
    # Main Delta-LL panels are rendered after global BH-FDR backfilling.

    # ─── PLOT 2: Coefficient bar charts (Cell 26 style) ───────────
    print("Generating coefficient bar charts...")
    from matplotlib.patches import Patch
    fig2_records = []

    for side in sides:
        for label in side2labels[side]:
            sig_list = significant_features[side][label]
            if not sig_list:
                print(f"  No significant features for {side} {label}, skipping coef plot.")
                continue
            row_size = len(sig_list)
            # Keep compact panels: no empty columns on left/right.
            plot_cols = row_size
            base_fig_w = plot_cols * 1.35
            fig_w = base_fig_w + COEF_EXTRA_RIGHT_INCH
            figsize = (fig_w, 4.2)
            fig, base_ax = plt.subplots(1, 1, figsize=figsize,
                                  gridspec_kw={'wspace': 0, 'hspace': 0}, sharex=True)
            base_ax.set_visible(False)
            row_id = 0
            for level in ['token', 'segment']:
                plt_index = row_id * plot_cols + 1
                for ft_i, ft in enumerate(sig_list):
                    ctrl_fts = control_features[side][level]
                    fts_for_model = ctrl_fts + [ft]
                    ax = plt.subplot(2, plot_cols, plt_index)
                    ax.set_box_aspect(COEF_AX_BOX_ASPECT)

                    key = f'control+{ft}'
                    if key not in all_results[side][level][label]:
                        plt_index += 1
                        continue

                    params = [sample['params'] for sample in all_results[side][level][label][key]
                              if sample is not None]
                    if not params:
                        plt_index += 1
                        continue

                    all_params = pd.DataFrame(params)
                    existing_fts = [f for f in fts_for_model if f in all_params.columns]
                    mean_coef = all_params[existing_fts].mean()
                    sem_coef = all_params[existing_fts].std() * 1.96 / 2
                    for ft_name in existing_fts:
                        fig2_records.append({
                            'side': side,
                            'label': label,
                            'level': level,
                            'focus_feature': ft,
                            'predictor': ft_name,
                            'mean_coef': mean_coef.get(ft_name, np.nan),
                            'ci95_halfwidth': sem_coef.get(ft_name, np.nan),
                            'n_folds': len(params),
                        })

                    n_bars = len(existing_fts)
                    bar_colors = (['tab:blue', 'tab:orange', 'tab:green'] + ['tab:red'])[:n_bars]
                    plt.bar(range(n_bars), mean_coef.values, yerr=sem_coef.values,
                            color=bar_colors, alpha=1, error_kw={'elinewidth': 3})
                    plt.tick_params(axis='both', which='major', labelsize=10)
                    plt.xticks([])
                    ax.yaxis.tick_right()
                    plt.title(feature_names.get(ft, ft), size=16)
                    if ft_i == 0:
                        ax.set_ylabel('word' if level == 'token' else level,
                                      labelpad=12, rotation=90, size=15)
                    plt.setp(ax.spines.values(), lw=1, color='lightgrey')
                    plt_index += 1
                row_id += 1

            legend_elements = [Patch(facecolor=color, edgecolor=color, label=lb)
                               for lb, color in zip(['length', 'frequency', 'position', 'predictor'],
                                                     ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])]
            if label == 'log_TrtS':
                # Keep legend inside figure canvas so exported height stays consistent.
                lgd = fig.legend(handles=legend_elements, loc='upper center',
                                 bbox_to_anchor=(0.5, COEF_LEGEND_Y), fontsize=12, ncol=4)
                left_frac = (0.06 * base_fig_w) / fig_w
                right_frac = (COEF_RIGHT_MARGIN * base_fig_w) / fig_w
                fig.subplots_adjust(left=left_frac, right=right_frac, bottom=0.08,
                                    top=COEF_TOP_MARGIN, wspace=0.35, hspace=0.35)
                plt.savefig(f'{FIGURES_DIR}/coef_bar_{label}.pdf', dpi=300)
                plt.savefig(f'{FIGURES_DIR}/coef_bar_{label}.png', dpi=300)
            else:
                # Reserve the same top margin as TrtS to keep subplot heights aligned.
                left_frac = (0.06 * base_fig_w) / fig_w
                right_frac = (COEF_RIGHT_MARGIN * base_fig_w) / fig_w
                fig.subplots_adjust(left=left_frac, right=right_frac, bottom=0.08,
                                    top=COEF_TOP_MARGIN, wspace=0.35, hspace=0.35)
                plt.savefig(f'{FIGURES_DIR}/coef_bar_{label}.pdf', dpi=300)
                plt.savefig(f'{FIGURES_DIR}/coef_bar_{label}.png', dpi=300)
            print(f"  Saved {FIGURES_DIR}/coef_bar_{label}.png")
            plt.close()
    figure_tables['figure2_coefficients'] = pd.DataFrame(fig2_records)

    # ─── PLOT 3: Surprisal-controlled delta-llh (Cell 19 style) ─────
    print("Generating surprisal-controlled delta-llh plot...")
    # Baseline with NMT-side surprisal control (source side uses LM surprisal; target side uses xent)
    surp_control_features = copy.deepcopy(control_features)
    surp_control_features['src'] = {level: surp_control_features['src'][level] + ['src_surp'] for level in levels}
    surp_control_features['tgt'] = {level: surp_control_features['tgt'][level] + ['xent'] for level in levels}
    # Baseline with LM-side surprisal control (source side uses src_surp; target side uses tgt_surp)
    lm_surp_control_features = copy.deepcopy(control_features)
    lm_surp_control_features['src'] = {level: lm_surp_control_features['src'][level] + ['src_surp'] for level in levels}
    lm_surp_control_features['tgt'] = {level: lm_surp_control_features['tgt'][level] + ['tgt_surp'] for level in levels}
    att_features_only = copy.deepcopy(features)
    att_features_only['src'] = [f for f in att_features_only['src'] if f != 'src_surp']
    att_features_only['tgt'] = [f for f in att_features_only['tgt'] if f not in ['tgt_surp', 'xent']]

    print("  Running cross-validation with surprisal control features...")
    surp_control_results = cross_validate(trains, tests, surp_control_features, att_features_only)
    print("  Running cross-validation with LM-side surprisal baselines...")
    lm_surp_control_results = cross_validate(trains, tests, lm_surp_control_features, att_features_only)

    add_surp_delta_alias(surp_control_results)
    add_surp_delta_alias(lm_surp_control_results)

    surp_sigfig = get_delta_sigfig(surp_control_results)
    lm_surp_sigfig = get_delta_sigfig(lm_surp_control_results)

    # For Figure 3, plot all attention features (not only significant ones).
    att_plot_features = {side: {} for side in sides}
    for side in sides:
        for label in side2labels[side]:
            att_plot_features[side][label] = list(att_features_only[side])

    llh_delta_surp_ctrl = get_metric_vals(surp_control_results, 'test_llh_delta', concat_values=True)
    llh_delta_lm_surp_ctrl = get_metric_vals(lm_surp_control_results, 'test_llh_delta', concat_values=True)

    # Also need delta against the original (non-surp) control
    llh_delta_vs_orig_ctrl = get_metric_vals(all_results, 'test_llh_delta', concat_values=True)

    print("  Running grouped cross-validation (task + language pair)...")
    grouped_results = {
        'all': {
            'status': 'ok',
            'results': all_results,
            'sigfig': sigfig,
            'llh_delta_ctrl': llh_delta_vs_orig_ctrl,
            'surp_results': surp_control_results,
            'surp_sigfig': surp_sigfig,
            'llh_delta_surp': llh_delta_surp_ctrl,
            'lm_surp_results': lm_surp_control_results,
            'lm_surp_sigfig': lm_surp_sigfig,
            'llh_delta_lm_surp': llh_delta_lm_surp_ctrl,
        }
    }

    for group_label in GROUP_LABELS:
        if group_label == 'all':
            continue

        group_data = filter_group_data(data, group_label)
        group_rows = len(group_data['src']['token'])
        group_translators = group_data['src']['token']['translator_id'].nunique() if group_rows > 0 else 0
        group_sentences = group_data['src']['token']['sentence_id'].nunique() if group_rows > 0 else 0
        print(f"    {group_label}: {group_rows} rows, {group_translators} translators, {group_sentences} sentences")

        gp_trains, gp_tests = get_train_test_split(group_data, n_splits=GROUP_N_SPLITS)
        if not gp_trains:
            grouped_results[group_label] = {'status': 'insufficient'}
            continue

        gp_results = cross_validate(gp_trains, gp_tests, control_features, features)
        gp_sigfig = get_delta_sigfig(gp_results)
        gp_llh_delta_ctrl = get_metric_vals(gp_results, 'test_llh_delta', concat_values=True)

        gp_surp_results = cross_validate(gp_trains, gp_tests, surp_control_features, att_features_only)
        add_surp_delta_alias(gp_surp_results)
        gp_surp_sigfig = get_delta_sigfig(gp_surp_results)
        gp_llh_delta_surp = get_metric_vals(gp_surp_results, 'test_llh_delta', concat_values=True)

        gp_lm_surp_results = cross_validate(gp_trains, gp_tests, lm_surp_control_features, att_features_only)
        add_surp_delta_alias(gp_lm_surp_results)
        gp_lm_surp_sigfig = get_delta_sigfig(gp_lm_surp_results)
        gp_llh_delta_lm_surp = get_metric_vals(gp_lm_surp_results, 'test_llh_delta', concat_values=True)

        grouped_results[group_label] = {
            'status': 'ok',
            'results': gp_results,
            'sigfig': gp_sigfig,
            'llh_delta_ctrl': gp_llh_delta_ctrl,
            'surp_results': gp_surp_results,
            'surp_sigfig': gp_surp_sigfig,
            'llh_delta_surp': gp_llh_delta_surp,
            'lm_surp_results': gp_lm_surp_results,
            'lm_surp_sigfig': gp_lm_surp_sigfig,
            'llh_delta_lm_surp': gp_llh_delta_lm_surp,
        }

    print("Running pooled interaction models...")
    task_interaction_results = cross_validate_interaction(
        trains, tests, control_features, features, interaction='task'
    )
    direction_interaction_results = cross_validate_interaction(
        trains, tests, control_features, features, interaction='direction'
    )
    task_interaction_sigfig = get_delta_sigfig(task_interaction_results)
    direction_interaction_sigfig = get_delta_sigfig(direction_interaction_results)
    task_interaction_delta = get_metric_vals(task_interaction_results, 'test_llh_delta', concat_values=True)
    direction_interaction_delta = get_metric_vals(direction_interaction_results, 'test_llh_delta', concat_values=True)

    interaction_sigfigs = {
        'task': task_interaction_sigfig,
        'direction': direction_interaction_sigfig,
    }

    # Global FDR over reported feature tests and pooled interaction tests.
    q_lookup, bh_scope_df = compute_global_q_lookup(
        grouped_results, att_features_only, interaction_sigfigs=interaction_sigfigs
    )
    figure_tables['table_bh_scope'] = bh_scope_df

    interaction_df = build_interaction_delta_table(
        {'task': task_interaction_delta, 'direction': direction_interaction_delta},
        interaction_sigfigs,
        q_lookup,
    )
    figure_tables['figure_interaction_delta'] = interaction_df
    figure_tables['table_interaction_top'] = build_interaction_top_table(interaction_df)

    # Backfill Figure 1 table with global-BH q-values (maps to group='all' control tests).
    if 'figure1_delta_llh_main' in figure_tables and not figure_tables['figure1_delta_llh_main'].empty:
        fig1_df = figure_tables['figure1_delta_llh_main'].copy()
        fig1_df['q_bh'] = fig1_df.apply(
            lambda r: q_lookup.get(('ctrl', 'all', r['side'], r['label'], r['level'], r['feature']), np.nan),
            axis=1
        )
        fig1_df['sig_bh'] = fig1_df['q_bh'] < 0.05
        figure_tables['figure1_delta_llh_main'] = fig1_df
        # Overwrite Figure 1 with FDR-based significance coloring.
        plot_main_delta_with_fdr(fig1_df)

    fig3_records = []
    row_ylim_att = {}
    for row_label in GROUP_LABELS:
        row_info = grouped_results.get(row_label, {'status': 'insufficient'})
        vals = []
        if row_info.get('status') == 'ok':
            ctrl_delta = row_info['llh_delta_ctrl']
            surp_delta = row_info['llh_delta_surp']
            for side in sides:
                for label in side2labels[side]:
                    for level_name in ['token', 'segment']:
                        ctrl_df = ctrl_delta[side][level_name][label].copy()
                        surp_df = surp_delta[side][level_name][label].copy()
                        for df_tmp in [ctrl_df, surp_df]:
                            if df_tmp.empty:
                                continue
                            df_tmp['feature'] = df_tmp['predictor'].apply(lambda x: x.split('+')[-1])
                            df_tmp = df_tmp[df_tmp['feature'].isin(att_plot_features[side][label])]
                            if not df_tmp.empty:
                                vals.extend(df_tmp['metric'].tolist())
        row_ylim_att[row_label] = compute_dynamic_ylim(vals)
    def plot_attention_grouped(row_layout, out_base):
        n_rows_plot = len(row_layout)
        legend_y = ATT_APPENDIX_LEGEND_Y if out_base.endswith('-split') else ATT_LEGEND_Y
        sep_ratio = GROUP_SPLIT_SEPARATOR_RATIO
        hspace_val = GROUP_MAIN_HSPACE
        fig, axes = plt.subplots(
            n_rows_plot, 3,
            figsize=(14.2, max(4.8, 0.85 * n_rows_plot + 1.8)),
            sharex='col',
            gridspec_kw={
                'width_ratios': [1, 1, 1],
                'height_ratios': row_height_ratios(row_layout, separator_ratio=sep_ratio),
                'wspace': 0.10,
                'hspace': hspace_val,
            }
        )
        if n_rows_plot == 1:
            axes = axes.reshape(1, -1)

        baseline_colors = {'*ctrl': '#5f7fa6', '*ctrl+surp': '#d39a62'}
        level_markers = {'token': 'o', 'segment': '^'}
        combo_order = [('token', '*ctrl'), ('token', '*ctrl+surp'),
                       ('segment', '*ctrl'), ('segment', '*ctrl+surp')]
        offsets = np.linspace(-0.30, 0.30, len(combo_order))

        for row_id, row_label in enumerate(row_layout):
            if row_label is None:
                for col_id in range(3):
                    axes[row_id][col_id].axis('off')
                continue

            row_info = grouped_results.get(row_label, {'status': 'insufficient'})
            for col_id, (side, label) in enumerate([(s, l) for s in sides for l in side2labels[s]]):
                ax = axes[row_id][col_id]
                if row_info.get('status') != 'ok':
                    ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                ctrl_delta = row_info['llh_delta_ctrl']
                surp_delta = row_info['llh_delta_surp']
                cur_sigfig = row_info['sigfig']
                cur_surp_sigfig = row_info['surp_sigfig']

                plot_parts = []
                for level_name in ['token', 'segment']:
                    base_df = ctrl_delta[side][level_name][label].copy()
                    base_df['feature'] = base_df['predictor'].apply(lambda x: x.split('+')[-1])
                    base_df['feature_id'] = base_df['feature'].apply(
                        lambda x: features[side].index(x) if x in features[side] else 999)
                    base_df = base_df.sort_values('feature_id')
                    base_df = base_df[base_df['feature'].isin(att_plot_features[side][label])]
                    base_df['level'] = level_name
                    base_df['delta'] = '*ctrl'
                    plot_parts.append(base_df)

                    surp_df = surp_delta[side][level_name][label].copy()
                    surp_df['feature'] = surp_df['predictor'].apply(lambda x: x.split('+')[-1])
                    surp_df['feature_id'] = surp_df['feature'].apply(
                        lambda x: features[side].index(x) if x in features[side] else 999)
                    surp_df = surp_df.sort_values('feature_id')
                    surp_df = surp_df[surp_df['feature'].isin(att_plot_features[side][label])]
                    surp_df['level'] = level_name
                    surp_df['delta'] = '*ctrl+surp'
                    plot_parts.append(surp_df)

                surp_df_all = pd.concat(plot_parts)
                if surp_df_all.empty:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                rec_df = surp_df_all.copy()
                rec_df['group'] = row_label
                rec_df['side'] = side
                rec_df['label'] = label
                rec_df['p_raw'] = rec_df.apply(
                    lambda r: (
                        (cur_surp_sigfig if r['delta'] == '*ctrl+surp' else cur_sigfig)
                        [side][r['level']][label][r['feature']].pvalue
                        if r['feature'] in (cur_surp_sigfig if r['delta'] == '*ctrl+surp' else cur_sigfig)[side][r['level']][label]
                        else np.nan
                    ),
                    axis=1
                )
                rec_df['n_participants'] = rec_df.apply(
                    lambda r: (
                        (cur_surp_sigfig if r['delta'] == '*ctrl+surp' else cur_sigfig)
                        [side][r['level']][label][r['feature']].n_participants
                        if r['feature'] in (cur_surp_sigfig if r['delta'] == '*ctrl+surp' else cur_sigfig)[side][r['level']][label]
                        else np.nan
                    ),
                    axis=1
                )
                rec_df['q_bh'] = rec_df.apply(
                    lambda r: q_lookup.get(
                        ('ctrl+surp' if r['delta'] == '*ctrl+surp' else 'ctrl',
                         r['group'], r['side'], r['label'], r['level'], r['feature']),
                        np.nan
                    ),
                    axis=1
                )
                rec_df['sig_bh'] = rec_df['q_bh'] < 0.05
                fig3_records.append(
                    rec_df[['group', 'side', 'label', 'level', 'delta', 'predictor',
                            'feature', 'metric', 'p_raw', 'n_participants', 'q_bh', 'sig_bh']]
                )

                surp_df_all['hue_col'] = surp_df_all[['level', 'delta']].apply(lambda x: ' '.join(x), axis=1)
                fts_in_plot = [ft for ft in features[side] if ft in set(surp_df_all['feature'])]
                x_positions = np.arange(len(fts_in_plot))

                for combo_idx, (level_name, delta_name) in enumerate(combo_order):
                    combo_df = surp_df_all[
                        (surp_df_all['level'] == level_name) &
                        (surp_df_all['delta'] == delta_name)
                    ]
                    for ft_idx, ft in enumerate(fts_in_plot):
                        ft_df = combo_df[combo_df['feature'] == ft]
                        if ft_df.empty:
                            continue
                        mean_y = ft_df['metric'].mean()
                        x_pos = x_positions[ft_idx] + offsets[combo_idx]
                        ax.scatter(
                            x_pos, mean_y,
                            color=baseline_colors[delta_name],
                            marker=level_markers[level_name],
                            s=44,
                            zorder=20,
                            edgecolors='none',
                        )
                        qval = q_lookup.get(
                            ('ctrl+surp' if delta_name == '*ctrl+surp' else 'ctrl',
                             row_label, side, label, level_name, ft),
                            np.nan
                        )
                        if bool(np.isfinite(qval) and qval < 0.05):
                            draw_significance_star(ax, x_pos, mean_y, SIG_STAR_BLACK_COLOR)

                ax.set_xlim(-0.55, len(fts_in_plot) - 0.45)
                ax.set_ylim(*attention_row_ylim(row_label, row_ylim_att, out_base))
                ax.axhline(0, ls='--', color=MAINDELTA_ZERO_LINE_COLOR, lw=MAINDELTA_ZERO_LINE_WIDTH)
                ax.grid(axis='y', color='0.94', linewidth=0.8)
                ax.set_ylabel('')
                ax.set_xlabel('')

                if col_id == 0:
                    ax.tick_params(axis='y', labelsize=ATT_YTICK_LABELSIZE)
                else:
                    ax.tick_params(axis='y', labelleft=False)
                if col_id == 2:
                    ax.set_ylabel(group_display_name(row_label), rotation=0, fontsize=11,
                                  ha='left', va='center', labelpad=28)
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.set_label_coords(ATT_ROW_LABEL_RIGHT_X, 0.5)

                if row_id == 0:
                    ax.set_title(label_names(side, None, label), size=13)

                if row_id == n_rows_plot - 1:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([feature_names.get(ft, ft) for ft in fts_in_plot],
                                       size=12, rotation=22, ha='right')
                else:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([])

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Token',
                   markerfacecolor='black', markeredgecolor='black', markersize=7),
            Line2D([0], [0], marker='^', color='w', label='Segment',
                   markerfacecolor='black', markeredgecolor='black', markersize=7),
            Line2D([0], [0], marker='o', color=baseline_colors['*ctrl'], label='Control only',
                   markerfacecolor=baseline_colors['*ctrl'], markeredgecolor=baseline_colors['*ctrl'],
                   linestyle='None', markersize=7),
            Line2D([0], [0], marker='o', color=baseline_colors['*ctrl+surp'], label='Control + surprisal',
                   markerfacecolor=baseline_colors['*ctrl+surp'], markeredgecolor=baseline_colors['*ctrl+surp'],
                   linestyle='None', markersize=7),
            Line2D([0], [0], marker='o', color='w', label='Significant',
                   markerfacecolor='black', markeredgecolor='black', linestyle='None', markersize=7),
            Line2D([0], [0], marker='o', color='w', label='Not significant',
                   markerfacecolor='white', markeredgecolor='black', linestyle='None', markersize=7),
        ]
        fig.text(0.02, 0.5, r'$\Delta$ llh per word/segment', va='center', rotation='vertical', size=12)
        lgd = fig.legend(handles=legend_elements, loc='lower center',
                         bbox_to_anchor=(0.5, legend_y), fontsize=11, ncol=6, frameon=True)
        fig.tight_layout(rect=[0.08, 0.08, 1.0, 1.0])
        plt.savefig(f'{FIGURES_DIR}/{out_base}.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        plt.savefig(f'{FIGURES_DIR}/{out_base}.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        print(f"  Saved {FIGURES_DIR}/{out_base}.png")
        plt.close()

    plot_attention_grouped(MAIN_ROW_LAYOUT, 'delta-llh-att')
    plot_attention_grouped(APPENDIX_ROW_LAYOUT, 'delta-llh-att-split')
    fig3_all_df = pd.concat(fig3_records, ignore_index=True) if fig3_records else pd.DataFrame()
    figure_tables['figure3_delta_llh_att_grouped'] = fig3_all_df[fig3_all_df['group'].isin(MAIN_SUMMARY_GROUPS)].copy() if not fig3_all_df.empty else pd.DataFrame()
    figure_tables['figure3_appendix_delta_llh_att_grouped_split'] = fig3_all_df[fig3_all_df['group'].isin(FULL_SPLIT_GROUPS)].copy() if not fig3_all_df.empty else pd.DataFrame()

    # ─── PLOT 4: Attention shrinkage summary (all sessions) ───────
    print("Generating attention shrinkage figure...")
    shrink_records = []
    all_row = grouped_results.get('all', {'status': 'insufficient'})
    if all_row.get('status') == 'ok':
        all_ctrl = all_row['llh_delta_ctrl']
        all_surp = all_row['llh_delta_surp']
        all_lm_surp = all_row['llh_delta_lm_surp']
        target_order = [('src', 'log_TrtS'), ('tgt', 'log_TrtT'), ('tgt', 'log_Dur')]
        level_order = ['token', 'segment']
        panel_labels = {
            ('src', 'log_TrtS', 'token'): 'Source reading (token)',
            ('src', 'log_TrtS', 'segment'): 'Source reading (segment)',
            ('tgt', 'log_TrtT', 'token'): 'Target reading (token)',
            ('tgt', 'log_TrtT', 'segment'): 'Target reading (segment)',
            ('tgt', 'log_Dur', 'token'): 'Target duration (token)',
            ('tgt', 'log_Dur', 'segment'): 'Target duration (segment)',
        }

        for side, label in target_order:
            for level_name in level_order:
                ctrl_df = all_ctrl[side][level_name][label].copy()
                nmt_df = all_surp[side][level_name][label].copy()
                lm_df = all_lm_surp[side][level_name][label].copy()
                att_cols = attention_features[side]
                if 'feature' in ctrl_df.columns:
                    ctrl_df = ctrl_df[ctrl_df['feature'].isin(att_cols)].copy()
                if 'feature' in nmt_df.columns:
                    nmt_df = nmt_df[nmt_df['feature'].isin(att_cols)].copy()
                if 'feature' in lm_df.columns:
                    lm_df = lm_df[lm_df['feature'].isin(att_cols)].copy()
                ctrl_mean = float(ctrl_df['metric'].mean()) if not ctrl_df.empty else np.nan
                nmt_mean = float(nmt_df['metric'].mean()) if not nmt_df.empty else np.nan
                lm_mean = float(lm_df['metric'].mean()) if not lm_df.empty else np.nan
                ctrl_vals = [float(v) for v in ctrl_df['metric'].dropna().tolist()] if not ctrl_df.empty else []
                lm_vals = [float(v) for v in lm_df['metric'].dropna().tolist()] if not lm_df.empty else []
                nmt_vals = [float(v) for v in nmt_df['metric'].dropna().tolist()] if not nmt_df.empty else []
                shrink_records.append({
                    'baseline_model': 'LM',
                    'side': side,
                    'target': label,
                    'level': level_name,
                    'cell': panel_labels[(side, label, level_name)],
                    'control_only': ctrl_mean,
                    'control_plus_surprisal': lm_mean,
                    'difference': ctrl_mean - lm_mean if np.isfinite(ctrl_mean) and np.isfinite(lm_mean) else np.nan,
                    'control_values_json': json.dumps(ctrl_vals, ensure_ascii=False),
                    'surprisal_values_json': json.dumps(lm_vals, ensure_ascii=False),
                })
                if side == 'src':
                    continue
                shrink_records.append({
                    'baseline_model': 'NMT',
                    'side': side,
                    'target': label,
                    'level': level_name,
                    'cell': panel_labels[(side, label, level_name)],
                    'control_only': ctrl_mean,
                    'control_plus_surprisal': nmt_mean,
                    'difference': ctrl_mean - nmt_mean if np.isfinite(ctrl_mean) and np.isfinite(nmt_mean) else np.nan,
                    'control_values_json': json.dumps(ctrl_vals, ensure_ascii=False),
                    'surprisal_values_json': json.dumps(nmt_vals, ensure_ascii=False),
                })

        shrink_df = pd.DataFrame(shrink_records)
        plot_att_shrinkage_from_table(shrink_df)
        figure_tables['figure4_att_shrinkage'] = shrink_df
    else:
        figure_tables['figure4_att_shrinkage'] = pd.DataFrame()

    # ─── PLOT 5: VIF heatmap ───────────────────────────────────────
    print("Generating VIF heatmap...")
    import scipy as sp

    def get_vif(df, cols):
        ck = np.column_stack([df[col].values for col in cols])
        cc = np.corrcoef(ck, rowvar=False)
        try:
            VIF = np.linalg.inv(cc)
            return VIF.diagonal().flatten()
        except np.linalg.LinAlgError:
            return np.ones(len(cols))

    fig, axes_vif = plt.subplots(2, 2, figsize=(8.2, 5.6), gridspec_kw={'wspace': 0.28, 'hspace': 0.55})
    fig4_records = []
    panel_file_stub = {
        ('src', 'token'): 'vif_src_token',
        ('src', 'segment'): 'vif_src_segment',
        ('tgt', 'token'): 'vif_tgt_token',
        ('tgt', 'segment'): 'vif_tgt_segment',
    }
    for rid, side in enumerate(sides):
        for cid, level in enumerate(['token', 'segment']):
            ax = axes_vif[rid, cid]
            df_vif = copy.deepcopy(data[side][level])
            fts_set = set(v for vals in significant_features[side].values() for v in vals)
            fts_list = [ft for ft in features[side] if ft in fts_set]
            if not fts_list:
                ax.set_visible(False)
                continue
            cols_needed = control_features[side][level] + fts_list
            df_vif = df_vif[cols_needed].dropna()
            if len(df_vif) < 10:
                ax.set_visible(False)
                continue
            corr_mtx = []
            for ft in fts_list:
                corrs = get_vif(df_vif, control_features[side][level] + [ft])
                corr_mtx.append(corrs)
                for vif_row_name, vif_val in zip(['length', 'frequency', 'position', 'predictor'], corrs):
                    fig4_records.append({
                        'side': side,
                        'level': level,
                        'feature': ft,
                        'vif_row': vif_row_name,
                        'vif_value': vif_val,
                    })
            corr_mtx = np.array(corr_mtx).T
            ft_labels = [feature_names.get(ft, ft) for ft in fts_list]
            sns.heatmap(corr_mtx, ax=ax, annot=True, annot_kws={"fontsize": 10},
                        cbar=None, cmap='Blues', vmax=2.3, vmin=1)
            ax.set_ylabel('')
            ax.set_yticklabels(['length', 'frequency', 'position', 'predictor'], rotation=0)
            ax.set_xticklabels(ft_labels, rotation=0, size=13)
            ax.xaxis.tick_top()

            fig_panel, ax_panel = plt.subplots(1, 1, figsize=(4.0, 2.6), constrained_layout=True)
            sns.heatmap(corr_mtx, ax=ax_panel, annot=True, annot_kws={"fontsize": 10},
                        cbar=None, cmap='Blues', vmax=2.3, vmin=1)
            ax_panel.set_ylabel('')
            ax_panel.set_yticklabels(['length', 'frequency', 'position', 'predictor'], rotation=0)
            ax_panel.set_xticklabels(ft_labels, rotation=0, size=12)
            ax_panel.xaxis.tick_top()
            stub = panel_file_stub[(side, level)]
            plt.savefig(f'{FIGURES_DIR}/{stub}.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(f'{FIGURES_DIR}/{stub}.png', bbox_inches='tight', dpi=300)
            print(f"  Saved {FIGURES_DIR}/{stub}.png")
            plt.close(fig_panel)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/vif.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{FIGURES_DIR}/vif.png', bbox_inches='tight', dpi=300)
    print(f"  Saved {FIGURES_DIR}/vif.png")
    plt.close()
    figure_tables['figure4_vif'] = pd.DataFrame(fig4_records)

    # ─── PLOT 6: Grouped delta-llh (all/task/language-pair) ──────
    print("Generating grouped delta-llh plot...")
    fig5_records = []
    row_ylim_grouped = {}
    for row_label in GROUP_LABELS:
        row_info = grouped_results.get(row_label, {'status': 'insufficient'})
        vals = []
        if row_info.get('status') == 'ok':
            row_llh = row_info['llh_delta_ctrl']
            for side in sides:
                for label in side2labels[side]:
                    for level_name in ['token', 'segment']:
                        df_tmp = row_llh[side][level_name][label]
                        if not df_tmp.empty:
                            vals.extend(df_tmp['metric'].tolist())
        row_ylim_grouped[row_label] = compute_dynamic_ylim(vals)

    def plot_grouped_delta(row_layout, out_base):
        n_rows_plot = len(row_layout)
        legend_y = GROUP_APPENDIX_LEGEND_Y if out_base.endswith('-split') else GROUP_LEGEND_Y
        fig, axes_lp = plt.subplots(
            n_rows_plot, 3,
            figsize=(14.0, max(4.8, 0.85 * n_rows_plot + 1.8)),
            sharex='col',
            gridspec_kw={
                'width_ratios': [1, 1, 1],
                'height_ratios': row_height_ratios(row_layout, separator_ratio=GROUP_SPLIT_SEPARATOR_RATIO),
                'wspace': 0.08,
                'hspace': GROUP_MAIN_HSPACE,
            }
        )
        if n_rows_plot == 1:
            axes_lp = axes_lp.reshape(1, -1)

        offsets = {'token': -0.10, 'segment': 0.10}
        markers = {'token': 'o', 'segment': '^'}

        for row_id, row_label in enumerate(row_layout):
            if row_label is None:
                for col_id in range(3):
                    axes_lp[row_id][col_id].axis('off')
                continue

            row_info = grouped_results.get(row_label, {'status': 'insufficient'})
            row_llh = row_info.get('llh_delta_ctrl')
            row_sigfig = row_info.get('sigfig')

            for col_id, (side, label) in enumerate([(s, l) for s in sides for l in side2labels[s]]):
                ax = axes_lp[row_id][col_id]

                if row_info.get('status') != 'ok':
                    ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                plot_parts = []
                for level_name in ['token', 'segment']:
                    df_level = row_llh[side][level_name][label].copy()
                    df_level['level'] = level_name
                    df_level['feature'] = df_level['predictor'].apply(lambda x: x.split('+')[-1])
                    plot_parts.append(df_level)

                df_combined = pd.concat(plot_parts)
                if df_combined.empty:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                df_combined['feature_id'] = df_combined['feature'].apply(
                    lambda x: features[side].index(x) if x in features[side] else 999)
                df_combined = df_combined.sort_values(['feature_id', 'level'])

                rec_df = df_combined.copy()
                rec_df['group'] = row_label
                rec_df['side'] = side
                rec_df['label'] = label
                rec_df['p_raw'] = rec_df.apply(
                    lambda r: (
                        row_sigfig[side][r['level']][label][r['feature']].pvalue
                        if r['feature'] in row_sigfig[side][r['level']][label] else np.nan
                    ),
                    axis=1
                )
                rec_df['n_participants'] = rec_df.apply(
                    lambda r: (
                        row_sigfig[side][r['level']][label][r['feature']].n_participants
                        if r['feature'] in row_sigfig[side][r['level']][label] else np.nan
                    ),
                    axis=1
                )
                rec_df['q_bh'] = rec_df.apply(
                    lambda r: q_lookup.get(('ctrl', r['group'], r['side'], r['label'], r['level'], r['feature']), np.nan),
                    axis=1
                )
                rec_df['sig_bh'] = rec_df['q_bh'] < 0.05
                fig5_records.append(
                    rec_df[['group', 'side', 'label', 'level', 'predictor', 'feature',
                            'metric', 'p_raw', 'n_participants', 'q_bh', 'sig_bh']]
                )

                fts = [ft for ft in features[side] if ft in set(df_combined['feature'])]
                x_positions = np.arange(len(fts))
                for level_name in ['token', 'segment']:
                    combo_df = df_combined[df_combined['level'] == level_name]
                    for ft_idx, ft in enumerate(fts):
                        ft_df = combo_df[combo_df['feature'] == ft]
                        if ft_df.empty:
                            continue
                        mean_y = ft_df['metric'].mean()
                        qval = q_lookup.get(('ctrl', row_label, side, label, level_name, ft), np.nan)
                        ax.scatter(
                            x_positions[ft_idx] + offsets[level_name],
                            mean_y,
                            color='#4a4a4a',
                            marker=markers[level_name],
                            s=42,
                            zorder=20,
                            edgecolors='none',
                        )
                        if bool(np.isfinite(qval) and qval < 0.05):
                            draw_significance_star(
                                ax, x_positions[ft_idx] + offsets[level_name], mean_y, SIG_STAR_WHITE_COLOR
                            )

                ax.set_xlim(-0.55, len(fts) - 0.45)
                ax.set_ylim(*grouped_row_ylim(row_label, row_ylim_grouped, out_base))
                ax.axhline(0, ls='--', color=MAINDELTA_ZERO_LINE_COLOR, lw=MAINDELTA_ZERO_LINE_WIDTH)
                ax.grid(axis='y', color='0.94', linewidth=0.8)
                ax.set_ylabel('')
                ax.set_xlabel('')

                if col_id == 0:
                    ax.tick_params(axis='y', labelsize=GROUP_YTICK_LABELSIZE)
                else:
                    ax.tick_params(axis='y', labelleft=False)
                if col_id == 2:
                    ax.set_ylabel(group_display_name(row_label), rotation=0, fontsize=11,
                                  ha='left', va='center', labelpad=28)
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.set_label_coords(GROUP_ROW_LABEL_RIGHT_X, 0.5)

                if row_id == 0:
                    ax.set_title(label_names(side, None, label, split=False), size=13)

                if row_id == n_rows_plot - 1:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([feature_names.get(ft, ft) for ft in fts], size=12, rotation=22, ha='right')
                else:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([])

        txt = fig.text(0.02, 0.5, r'$\Delta$ llh against control features', va='center', rotation='vertical', size=12)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Token',
                   markerfacecolor='black', markeredgecolor='black', markersize=7),
            Line2D([0], [0], marker='^', color='w', label='Segment',
                   markerfacecolor='black', markeredgecolor='black', markersize=7),
            Line2D([0], [0], marker='o', color='w', label='Significant',
                   markerfacecolor='black', markeredgecolor='black', linestyle='None', markersize=7),
            Line2D([0], [0], marker='o', color='w', label='Not significant',
                   markerfacecolor='white', markeredgecolor='black', linestyle='None', markersize=7),
        ]
        lgd = fig.legend(handles=legend_elements, loc='lower center',
                         bbox_to_anchor=(0.5, legend_y), fontsize=11, ncol=4, frameon=True)
        fig.tight_layout(rect=[0.08, 0.08, 1.0, 1.0])
        plt.savefig(f'{FIGURES_DIR}/{out_base}.pdf',
                    bbox_extra_artists=(lgd, txt), bbox_inches='tight', dpi=300)
        plt.savefig(f'{FIGURES_DIR}/{out_base}.png',
                    bbox_extra_artists=(lgd, txt), bbox_inches='tight', dpi=300)
        print(f"  Saved {FIGURES_DIR}/{out_base}.png")
        plt.close()

    plot_grouped_delta(MAIN_ROW_LAYOUT, 'delta-llh-lang_pairs')
    plot_grouped_delta(APPENDIX_ROW_LAYOUT, 'delta-llh-lang_pairs-split')
    fig5_all_df = pd.concat(fig5_records, ignore_index=True) if fig5_records else pd.DataFrame()
    figure_tables['figure5_delta_llh_grouped'] = fig5_all_df[fig5_all_df['group'].isin(MAIN_SUMMARY_GROUPS)].copy() if not fig5_all_df.empty else pd.DataFrame()
    figure_tables['figure5_appendix_delta_llh_grouped_split'] = fig5_all_df[fig5_all_df['group'].isin(FULL_SPLIT_GROUPS)].copy() if not fig5_all_df.empty else pd.DataFrame()

    # ─── PLOT 6 & 7: POS pointplots (Cells 43 & 44 style) ────────
    print("Generating POS pointplots...")
    # We need POS tags. Use spacy for en tokens.
    has_nltk = False
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        has_spacy = True
    except Exception:
        has_spacy = False
        print("  spaCy not available, trying nltk...")
        try:
            import nltk
            has_nltk = True
        except Exception:
            has_nltk = False

    src_token_df = copy.deepcopy(data['src']['token'])

    def get_pos(token):
        if has_spacy:
            doc = nlp(str(token))
            return doc[0].pos_ if doc else 'X'
        elif has_nltk:
            try:
                tags = nltk.pos_tag([str(token)], tagset='universal')
                return tags[0][1] if tags else 'X'
            except Exception:
                return 'X'
        return 'X'

    if has_spacy or has_nltk:
        # Get unique tokens and POS tag them
        unique_tokens = src_token_df['src_tokens'].unique()
        token_pos_map = {}
        for tok in unique_tokens:
            token_pos_map[tok] = get_pos(tok)
        src_token_df['POS'] = src_token_df['src_tokens'].map(token_pos_map)

        # --- src POS plot (Cell 43) ---
        src_sig_fts = significant_features['src'].get('log_TrtS', [])
        src_pos_records = []
        if src_sig_fts:
            fts = ['log_TrtS'] + src_sig_fts
            fig, axes_pos = plt.subplots(1, len(fts), figsize=(min(9, len(fts) * 2.5), 3.5),
                                         sharey=True, sharex=False, constrained_layout=True,
                                         gridspec_kw={'wspace': 0.0})
            if len(fts) == 1:
                axes_pos = [axes_pos]
            df_pos = src_token_df[src_token_df['log_TrtS'].notnull()].copy()
            # Filter POS categories with enough data
            pos_counts = df_pos['POS'].value_counts()
            valid_pos = pos_counts[pos_counts >= 20].index
            df_pos = df_pos[df_pos['POS'].isin(valid_pos)]
            tofunc_order = list(df_pos[['POS', 'log_TrtS']].groupby('POS').mean().sort_values('log_TrtS').index)

            for i, (ax, ft) in enumerate(zip(axes_pos, fts)):
                if ft not in df_pos.columns:
                    continue
                new_df = pd.DataFrame({'pos': df_pos.POS, ft: df_pos[ft].values})
                new_df['rank'] = new_df['pos'].apply(lambda x: tofunc_order.index(x) if x in tofunc_order else 999)
                new_df = new_df.sort_values(by='rank')
                src_pos_records.append(new_df.assign(feature=ft)[['pos', 'feature', ft]].rename(columns={ft: 'value'}))
                panel_color = POS_PRIMARY_COLOR if i == 0 else POS_SECONDARY_COLOR
                sns.pointplot(data=new_df, x=ft, y='pos', ax=ax, capsize=.0, join=False, color=panel_color, scale=0.9)
                for line in ax.lines:
                    line.set_color(POS_ERR_COLOR)
                    line.set_linewidth(1.0)
                ax.tick_params(axis='y', which='major', labelsize=POS_TICK_SIZE)
                if i > 0:
                    ax.get_yaxis().set_visible(False)
                    ax.set_xlabel(feature_names.get(ft, 'src reading time'), size=POS_LABEL_SIZE)
                    pcorr = pearsonr(df_pos[ft].dropna(), df_pos.loc[df_pos[ft].notnull(), 'log_TrtS'])
                    pstr = r'$\rho=$' + f'${pcorr.statistic:.2f}^*$' if pcorr.pvalue < .001 else r'$\rho=$' + f'${pcorr.statistic:.2f}$'
                    ax.set_title(pstr, size=POS_TITLE_SIZE, color=POS_PRIMARY_COLOR)
                else:
                    ax.set_ylabel('')
                    ax.set_xlabel(feature_names.get(ft, 'src reading\ntime'), size=POS_LABEL_SIZE)
                for spine in ax.spines.values():
                    spine.set_color(POS_SPINE_COLOR)
                    spine.set_linewidth(POS_SPINE_WIDTH)

            plt.savefig(f'{FIGURES_DIR}/src-POS.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(f'{FIGURES_DIR}/src-POS.png', bbox_inches='tight', dpi=300)
            print(f"  Saved {FIGURES_DIR}/src-POS.png")
            plt.close()
        figure_tables['figure6_src_pos'] = pd.concat(src_pos_records, ignore_index=True) if src_pos_records else pd.DataFrame()

        # --- tgt POS plot (Cell 44) ---
        tgt_token_df = copy.deepcopy(data['tgt']['token'])
        # For Chinese target, POS tagging is different. Try jieba/spacy-zh if available.
        tgt_lang = tgt_token_df['tgt_lang'].iloc[0] if len(tgt_token_df) > 0 else 'zh'
        tgt_sig_fts = significant_features['tgt'].get('log_Dur', [])
        tgt_pos_records = []

        if tgt_sig_fts:
            try:
                if tgt_lang.startswith('zh'):
                    import jieba.posseg as pseg
                    def get_tgt_pos(token):
                        words = pseg.cut(str(token))
                        for w, flag in words:
                            return flag.upper()
                        return 'X'
                else:
                    get_tgt_pos = get_pos

                unique_tgt_tokens = tgt_token_df['tgt_tokens'].unique()
                tgt_token_pos_map = {}
                for tok in unique_tgt_tokens:
                    tgt_token_pos_map[tok] = get_tgt_pos(tok)
                tgt_token_df['POS'] = tgt_token_df['tgt_tokens'].map(tgt_token_pos_map)

                fts = ['log_Dur', 'xent'] + [ft for ft in tgt_sig_fts if ft not in ['xent', 'tgt_surp']]
                fig, axes_tpos = plt.subplots(1, len(fts), figsize=(min(15, len(fts) * 2.5), 3.5),
                                              sharex=False, sharey=True,
                                              constrained_layout=True, gridspec_kw={'wspace': 0.1})
                if len(fts) == 1:
                    axes_tpos = [axes_tpos]

                df_tpos = tgt_token_df[tgt_token_df['log_Dur'].notnull()].copy()
                pos_counts = df_tpos['POS'].value_counts()
                valid_pos = pos_counts[pos_counts >= 20].index
                df_tpos = df_tpos[df_tpos['POS'].isin(valid_pos)]
                tofunc_order = list(df_tpos[['POS', 'log_Dur']].groupby('POS').mean().sort_values('log_Dur').index)

                for row_idx, ft in enumerate(fts):
                    ax = axes_tpos[row_idx]
                    if ft not in df_tpos.columns:
                        continue
                    new_df = pd.DataFrame({'pos': df_tpos.POS, ft: df_tpos[ft].values})
                    new_df['rank'] = new_df['pos'].apply(lambda x: tofunc_order.index(x) if x in tofunc_order else 999)
                    new_df = new_df.sort_values(by=['rank'])
                    tgt_pos_records.append(new_df.assign(feature=ft)[['pos', 'feature', ft]].rename(columns={ft: 'value'}))
                    panel_color = POS_PRIMARY_COLOR if row_idx == 0 else POS_SECONDARY_COLOR
                    sns.pointplot(data=new_df, x=ft, y='pos', ax=ax, capsize=.0, join=False, color=panel_color, scale=0.9)
                    for line in ax.lines:
                        line.set_color(POS_ERR_COLOR)
                        line.set_linewidth(1.0)
                    ax.tick_params(axis='y', which='major', labelsize=POS_TICK_SIZE)
                    if row_idx > 0:
                        ax.get_yaxis().set_visible(False)
                        ax.set_xlabel(feature_names.get(ft, 'tgt\nduration'), size=POS_LABEL_SIZE)
                        pcorr = pearsonr(df_tpos[ft].dropna(), df_tpos.loc[df_tpos[ft].notnull(), 'log_Dur'])
                        pstr = r'$\rho=$' + f'${pcorr.statistic:.2f}^*$' if pcorr.pvalue < .001 else r'$\rho=$' + f'${pcorr.statistic:.2f}$'
                        ax.set_title(pstr, size=POS_TITLE_SIZE, color=POS_PRIMARY_COLOR)
                    else:
                        ax.set_ylabel('')
                        ax.set_xlabel(feature_names.get(ft, 'tgt\nduration'), size=POS_LABEL_SIZE)
                    for spine in ax.spines.values():
                        spine.set_color(POS_SPINE_COLOR)
                        spine.set_linewidth(POS_SPINE_WIDTH)

                plt.tight_layout()
                plt.savefig(f'{FIGURES_DIR}/tgt-POS.pdf', bbox_inches='tight', dpi=300)
                plt.savefig(f'{FIGURES_DIR}/tgt-POS.png', bbox_inches='tight', dpi=300)
                print(f"  Saved {FIGURES_DIR}/tgt-POS.png")
                plt.close()
                figure_tables['figure7_tgt_pos'] = pd.concat(tgt_pos_records, ignore_index=True) if tgt_pos_records else pd.DataFrame()
            except Exception as e:
                print(f"  tgt POS plot failed: {e}")
                figure_tables['figure7_tgt_pos'] = pd.DataFrame()
    else:
        print("  Skipping POS plots (no POS tagger available)")
        figure_tables['figure6_src_pos'] = pd.DataFrame()
        figure_tables['figure7_tgt_pos'] = pd.DataFrame()

    # ─── PLOT 8: Example sentence (Cell 45 style) ────────────────
    print("Generating example sentence plot...")
    tgt_data_raw = copy.deepcopy(data['tgt']['token'])
    # Pick a sentence with enough data points
    seq_counts = tgt_data_raw.groupby('src_seq_id').size()
    good_seqs = seq_counts[seq_counts >= 20].index
    if len(good_seqs) > 0:
        sent = good_seqs[len(good_seqs) // 2]  # pick a middle one
        fts_ex = ['log_Dur', 'xent', 'y2srcseq_ent']
        ft_name_ex = {'log_Dur': 'translation\nduration', 'xent': 'translation\nsurprisal',
                      'y2srcseq_ent': 'cross-attention\nentropy'}
        fig, axes_ex = plt.subplots(1, len(fts_ex), figsize=(4, 4.2), sharey=True,
                                    constrained_layout=True, gridspec_kw={'wspace': 0.1})
        df_ex = tgt_data_raw[(tgt_data_raw['log_Dur'].notnull()) & (tgt_data_raw['src_seq_id'] == sent)].copy()
        if len(df_ex) > 3:
            df_ex['src_mean_id'] = df_ex['src_token_ids'].apply(
                lambda x: np.mean([int(i) for i in str(x).split('+') if i.isdigit()]))
            df_ex = df_ex.sort_values(['src_mean_id'])[['src_tokens'] + fts_ex]
            figure_tables['figure8_example_sentence'] = df_ex.copy()

            for i, ft in enumerate(fts_ex):
                ax = axes_ex[i]
                if ft not in df_ex.columns:
                    continue
                new_df = pd.DataFrame({'src_tokens': df_ex.src_tokens, 'values': df_ex[ft].values})
                sns.pointplot(data=new_df, y='src_tokens', x='values', color=EXAMPLE_LINE_COLOR,
                              dodge=True, join=False, ax=ax, scale=0.6)
                sns.pointplot(data=new_df, y='src_tokens', x='values', color=POS_PRIMARY_COLOR,
                              ax=ax, errorbar=None, join=False, scale=0.7)
                plt.setp(ax.collections, zorder=100, label="")
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='major', labelsize=EXAMPLE_TOKEN_SIZE)
                if i > 0:
                    ax.get_yaxis().set_visible(False)
                    pcorr = pearsonr(df_ex[ft].dropna(), df_ex.loc[df_ex[ft].notnull(), 'log_Dur'])
                    pstr = r'$\rho=$' + f'${pcorr.statistic:.2f}^*$' if pcorr.pvalue < .05 else r'$\rho=$' + f'${pcorr.statistic:.2f}$'
                    ax.set_title(pstr, size=EXAMPLE_TITLE_SIZE, color=POS_PRIMARY_COLOR)
                ax.set_xlabel(ft_name_ex.get(ft, ft), size=10)
                for spine in ax.spines.values():
                    spine.set_color(POS_SPINE_COLOR)
                    spine.set_linewidth(POS_SPINE_WIDTH)

            plt.tight_layout()
            plt.savefig(f'{FIGURES_DIR}/tgt-example.pdf', bbox_inches='tight', dpi=300, pad_inches=0.05)
            plt.savefig(f'{FIGURES_DIR}/tgt-example.png', bbox_inches='tight', dpi=300, pad_inches=0.05)
            print(f"  Saved {FIGURES_DIR}/tgt-example.png")
            plt.close()
        else:
            print("  Not enough data for example sentence plot")
            figure_tables['figure8_example_sentence'] = pd.DataFrame()
    else:
        print("  No suitable sentence found for example plot")
        figure_tables['figure8_example_sentence'] = pd.DataFrame()

    # ─── TABLES: Repeated-CV stability + effect sizes ─────────────
    stability_df, stability_series, participant_series = build_stability_table(
        data, repeats=STABILITY_REPEATS, n_splits=GROUP_N_SPLITS
    )
    figure_tables['table_stability'] = stability_df

    effect_df = build_effect_size_table(participant_series, bootstrap_n=BOOTSTRAP_N)
    figure_tables['table_effect_sizes'] = effect_df

    dump_figure_data(FIGURES_DIR, figure_tables)
    if render_after:
        render_figures_from_saved_tables()
    print("Done! All figures saved to", FIGURES_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce Yiyu analysis or redraw figures from saved TSV outputs."
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run analysis and refresh TSV outputs without redrawing figures.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip statistical recomputation and redraw figures from the existing TSV files in figures_yiyu/.",
    )
    parser.add_argument(
        "--fig",
        action="append",
        choices=sorted(PLOT_KEYS),
        help="Redraw only the selected figure(s) from saved TSV files. Implies --plots-only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_figs = set(args.fig) if args.fig else None
    if args.plots_only or selected_figs is not None:
        render_figures_from_saved_tables(selected_figs=selected_figs)
    else:
        main(render_after=not args.analysis_only)
