"""Export the trained linear Pipeline (imputers + scaler + one-hot + Ridge / Logistic Regression)
to model.json, so the static web app (index.html + model.js) can make the same predictions in the browser.

Run from the task folder:  python static_app/export_model.py
"""
import glob
import json
import os

import joblib
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
(model_path,) = glob.glob(os.path.join(os.path.dirname(here), 'models', '*.pkl'))
model = joblib.load(model_path)
pre, est = model.steps[0][1], model.steps[-1][1]

export = {'num_cols': [], 'num_median': [], 'num_mean': [], 'num_scale': [],
          'cat_cols': [], 'cat_fill': [], 'cat_categories': [], 'cat_drop': []}

for name, pipe, cols in pre.transformers_:
    if pipe == 'drop':
        continue
    steps = pipe.named_steps
    if 'scaler' in steps:                                  # numbers: median imputer + StandardScaler
        export['num_cols'] += list(cols)
        export['num_median'] += steps['imputer'].statistics_.tolist()
        export['num_mean'] += steps['scaler'].mean_.tolist()
        export['num_scale'] += steps['scaler'].scale_.tolist()
    else:                                                  # text: most-frequent imputer + OneHotEncoder
        enc = steps['encoder']
        assert enc.handle_unknown == 'ignore'
        drop = enc.drop_idx_ if enc.drop_idx_ is not None else [None] * len(cols)
        for j, col in enumerate(cols):
            export['cat_cols'].append(col)
            export['cat_fill'].append(str(steps['imputer'].statistics_[j]))
            export['cat_categories'].append([str(c) for c in enc.categories_[j]])
            export['cat_drop'].append(-1 if drop[j] is None else int(drop[j]))

# the coefficients are in the same order as the preprocessor output: numbers first, then one-hot columns
n_out = len(model[:-1].get_feature_names_out())
coef = np.ravel(est.coef_)
assert len(coef) == n_out
export['coef'] = coef.tolist()
export['intercept'] = float(np.ravel(est.intercept_)[0])
export['kind'] = 'logistic' if hasattr(est, 'predict_proba') else 'regression'
if export['kind'] == 'logistic':
    export['classes'] = [int(c) for c in est.classes_]

path = os.path.join(here, 'model.json')
with open(path, 'w') as f:
    json.dump(export, f, indent=1)
print(f'saved {path} ({export["kind"]}, {n_out} inputs to the model)')
