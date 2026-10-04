function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function PredictionBadge({ value }) {
  return (
    <span className={`badge ${value ? 'badge-positive' : 'badge-negative'}`}>
      {value ? '\u25CF Positive' : '\u25CB Negative'}
    </span>
  )
}

export default function PredictionResults({ predictions, onExplain, isExplaining }) {
  if (!predictions) return null

  const rows = Object.entries(predictions)

  return (
    <section className="card">
      <h2>Predictions</h2>
      <div className="table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th>Label</th>
              <th>E04 probability</th>
              <th>E04 prediction</th>
              <th>E08 probability</th>
              <th>E08 prediction</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, item]) => (
              <tr key={label}>
                <td className="label-name">{label}</td>
                <td>{formatPercent(item.e04_probability)}</td>
                <td>
                  <PredictionBadge value={item.e04_prediction} />
                </td>
                <td>{formatPercent(item.e08_probability)}</td>
                <td>
                  <PredictionBadge value={item.e08_prediction} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button type="button" className="button primary" onClick={onExplain} disabled={isExplaining}>
        {isExplaining ? 'Explaining…' : 'Explain'}
      </button>
    </section>
  )
}
