// Same prediction as the scikit-learn Pipeline in ../models/*.pkl,
// using the numbers exported by export_model.py (model.json).
(function (root) {
  function linearScore(model, row) {
    var z = model.intercept, k = 0, i, j, c;

    // numbers: median imputer -> StandardScaler
    for (i = 0; i < model.num_cols.length; i++, k++) {
      var v = row[model.num_cols[i]];
      if (v === undefined || v === null || isNaN(v)) v = model.num_median[i];
      z += model.coef[k] * (v - model.num_mean[i]) / model.num_scale[i];
    }

    // text: most-frequent imputer -> OneHotEncoder (dropped column skipped, unknown -> all zeros)
    for (j = 0; j < model.cat_cols.length; j++) {
      var value = row[model.cat_cols[j]];
      if (value === undefined || value === null || value === '') value = model.cat_fill[j];
      var cats = model.cat_categories[j], drop = model.cat_drop[j];
      for (c = 0; c < cats.length; c++) {
        if (c === drop) continue;
        if (cats[c] === String(value)) z += model.coef[k];
        k++;
      }
    }
    return z;
  }

  // regression: the predicted value; logistic: probability of classes[1]
  function predict(model, row) {
    var z = linearScore(model, row);
    return model.kind === 'logistic' ? 1 / (1 + Math.exp(-z)) : z;
  }

  var api = { predict: predict };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LinearModel = api;
})(this);
