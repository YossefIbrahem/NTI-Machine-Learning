// Same prediction as the scikit-learn Pipeline in models/house_price_model.pkl,
// using the numbers exported by export_model.py (model.json).
(function (root) {
  function predictLogPrice(model, house) {
    // same feature engineering as the notebook / app.py
    var features = {
      Bedrooms: house.Bedrooms,
      Bathrooms: house.Bathrooms,
      Area: house.Area,
      Log_Area: Math.log1p(house.Area),
      Area_per_Bedroom: house.Area / house.Bedrooms,
    };

    var nNum = model.num_cols.length;
    var x = new Array(nNum);
    for (var k = 0; k < nNum; k++) {
      var v = features[model.num_cols[k]];
      if (v === undefined || v === null || isNaN(v)) v = model.num_median[k];
      x[k] = (v - model.num_mean[k]) / model.num_scale[k];
    }

    var nCat = model.cat_cols.length;
    var xc = new Array(nCat);
    for (var j = 0; j < nCat; j++) {
      var value = house[model.cat_cols[j]];
      if (value === undefined || value === null || value === '') value = 'Unknown';
      var idx = model.cat_maps[j][String(value)];
      xc[j] = idx === undefined ? model.cat_unknown[j] : idx;
    }

    // RBF-kernel SVR: sum(dual * exp(-gamma * ||x - sv||^2)) + intercept
    var total = model.intercept;
    var n = model.dual.length;
    for (var i = 0; i < n; i++) {
      var d = 0;
      for (var a = 0; a < nNum; a++) {
        var diff = x[a] - model.sv_num[i * nNum + a];
        d += diff * diff;
      }
      for (var b = 0; b < nCat; b++) {
        var s = model.sv_cat[i * nCat + b];
        if (xc[b] !== s) d += (xc[b] >= 0 ? 1 : 0) + (s >= 0 ? 1 : 0);
      }
      total += model.dual[i] * Math.exp(-model.gamma * d);
    }
    return total;
  }

  function predictPrice(model, house) {
    return Math.expm1(predictLogPrice(model, house));   // the model predicts log(Price)
  }

  var api = { predictLogPrice: predictLogPrice, predictPrice: predictPrice };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.HouseModel = api;
})(this);
