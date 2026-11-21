"""Export the trained Pipeline (preprocessing + SVR) to model.json for the static web app.

The web page (index.html + model.js) recomputes exactly what the Pipeline does:
  numbers  -> (x - mean) / scale
  text     -> one-hot index (rare compounds / cities share the "infrequent" column)
  SVR      -> sum(dual_coef * exp(-gamma * ||x - sv||^2)) + intercept   (= log price)

Run from the Final_Project folder:  python static_app/export_model.py
"""
import json
import os
import warnings

import joblib
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
project = os.path.dirname(here)

model = joblib.load(os.path.join(project, 'models', 'house_price_model.pkl'))
with open(os.path.join(project, 'models', 'choices.json')) as f:
    choices = json.load(f)

pre, svr = model.named_steps['preprocessor'], model.named_steps['model']
assert svr.kernel == 'rbf'

# ---- numeric part: median imputer + StandardScaler ----
num_pipe = pre.named_transformers_['num']
num_cols = list(pre.transformers_[0][2])
imputer, scaler = num_pipe.named_steps['imputer'], num_pipe.named_steps['scaler']

# ---- categorical part: 'Unknown' imputer + OneHotEncoder(min_frequency=10) ----
cat_cols = list(pre.transformers_[1][2])
encoder = pre.named_transformers_['cat'].named_steps['encoder']
sizes = list(encoder._n_features_outs)
offsets = np.cumsum([0] + sizes[:-1])


def local_index(feature, value):
    """Column (inside this feature's block) that the encoder sets to 1, or -1 if none."""
    row = np.array([[encoder.categories_[j][0] for j in range(len(cat_cols))]], dtype=object)
    row[0, feature] = value
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # unseen value -> "infrequent" column (expected)
        out = encoder.transform(row).toarray()[0]
    block = out[offsets[feature]:offsets[feature] + sizes[feature]]
    hits = np.flatnonzero(block)
    return int(hits[0]) if len(hits) else -1


cat_maps, cat_unknown = [], []
for j, col in enumerate(cat_cols):
    cat_maps.append({str(v): local_index(j, v) for v in encoder.categories_[j]})
    cat_unknown.append(local_index(j, '__never_seen__'))

# ---- support vectors: 5 dense numbers + one local index per categorical block ----
sv = svr.support_vectors_.tocsr()
n_num = len(num_cols)
sv_num = sv[:, :n_num].toarray()
sv_cat = np.full((sv.shape[0], len(cat_cols)), -1, dtype=int)
cat_block = sv[:, n_num:].tocsr()
for i in range(sv.shape[0]):
    for col in cat_block.indices[cat_block.indptr[i]:cat_block.indptr[i + 1]]:
        j = int(np.searchsorted(offsets, col, side='right') - 1)
        sv_cat[i, j] = int(col - offsets[j])
assert set(np.unique(cat_block.data)) == {1.0}

export = {
    'gamma': float(svr._gamma),
    'intercept': float(svr.intercept_[0]),
    'num_cols': num_cols,
    'num_median': imputer.statistics_.tolist(),
    'num_mean': scaler.mean_.tolist(),
    'num_scale': scaler.scale_.tolist(),
    'cat_cols': cat_cols,
    'cat_maps': cat_maps,
    'cat_unknown': cat_unknown,
    'choices': choices,
    'dual': np.asarray(svr.dual_coef_.todense() if hasattr(svr.dual_coef_, 'todense') else svr.dual_coef_)[0].tolist(),
    'sv_num': sv_num.ravel().tolist(),
    'sv_cat': sv_cat.ravel().tolist(),
}
path = os.path.join(here, 'model.json')
with open(path, 'w') as f:
    json.dump(export, f, separators=(',', ':'))
print(f'saved {path}  ({os.path.getsize(path) / 1e6:.2f} MB, {sv.shape[0]} support vectors)')
