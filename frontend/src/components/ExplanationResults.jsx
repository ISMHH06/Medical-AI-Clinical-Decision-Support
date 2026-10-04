function ShapBarList({ features }) {
  const maxAbs = Math.max(
    ...features.map((feature) => Math.abs(feature.shap_value)),
    Number.EPSILON,
  )

  return (
    <div className="shap-list">
      {features.map((feature) => {
        const magnitude = Math.abs(feature.shap_value)
        const widthPercent = (magnitude / maxAbs) * 100
        const isPositive = feature.shap_value >= 0
        return (
          <div className="shap-row" key={feature.feature_name}>
            <span className="shap-name" title={feature.feature_name}>
              {feature.feature_name}
            </span>
            <span className="shap-track">
              <span
                className={`shap-bar ${isPositive ? 'shap-positive' : 'shap-negative'}`}
                style={{ width: `${widthPercent}%` }}
              />
            </span>
            <span className={`shap-value ${isPositive ? 'text-positive' : 'text-negative'}`}>
              {feature.shap_value >= 0 ? '+' : ''}
              {feature.shap_value.toFixed(3)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export default function ExplanationResults({ explanations }) {
  if (!explanations) return null

  return (
    <section className="card">
      <h2>Explanations</h2>
      <div className="explanation-grid">
        {Object.entries(explanations).map(([label, item]) => (
          <div className="explanation-card" key={label}>
            <div className="explanation-header">
              <h3>{label}</h3>
              <span className="explanation-prob">
                {(item.calibrated_probability * 100).toFixed(1)}%
              </span>
            </div>
            <img
              className="gradcam-image"
              src={`data:image/png;base64,${item.gradcam_image_base64}`}
              alt={`Grad-CAM heatmap for ${label}`}
            />
            <div className="shap-section">
              <h4>Top SHAP features</h4>
              <ShapBarList features={item.top_shap_features} />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
