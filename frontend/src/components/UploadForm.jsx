const SEX_OPTIONS = ['Female', 'Male', 'Unknown']

const RACE_OPTIONS = [
  'Asian',
  'Black',
  'Native American',
  'Other',
  'Pacific Islander',
  'Patient Refused',
  'Unknown',
  'White',
]

const ETHNICITY_OPTIONS = [
  'Hispanic/Latino',
  'Non-Hispanic/Non-Latino',
  'Patient Refused',
  'Unknown',
]

const INSURANCE_OPTIONS = ['Medicaid', 'Medicare', 'Other', 'Private Insurance', 'Unknown']

function SelectField({ id, label, value, options, onChange }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  )
}

export default function UploadForm({
  imageFile,
  onImageChange,
  fields,
  onFieldChange,
  onPredict,
  isPredicting,
}) {
  const handleSubmit = (event) => {
    event.preventDefault()
    onPredict({ imageFile, ...fields })
  }

  return (
    <section className="card">
      <h2>Input</h2>
      <form onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="image">Chest X-ray image</label>
          <input
            id="image"
            type="file"
            accept="image/*"
            onChange={(event) => onImageChange(event.target.files?.[0] ?? null)}
          />
          {imageFile ? <span className="file-name">{imageFile.name}</span> : null}
        </div>

        <div className="form-grid">
          <div className="field">
            <label htmlFor="age">Age *</label>
            <input
              id="age"
              type="number"
              min="0"
              max="120"
              step="1"
              required
              value={fields.age}
              onChange={(event) => onFieldChange('age', event.target.value)}
            />
          </div>

          <SelectField
            id="sex"
            label="Sex"
            value={fields.sex}
            options={SEX_OPTIONS}
            onChange={(value) => onFieldChange('sex', value)}
          />

          <SelectField
            id="race"
            label="Race"
            value={fields.race}
            options={RACE_OPTIONS}
            onChange={(value) => onFieldChange('race', value)}
          />

          <SelectField
            id="ethnicity"
            label="Ethnicity"
            value={fields.ethnicity}
            options={ETHNICITY_OPTIONS}
            onChange={(value) => onFieldChange('ethnicity', value)}
          />

          <SelectField
            id="insurance_type"
            label="Insurance type"
            value={fields.insurance_type}
            options={INSURANCE_OPTIONS}
            onChange={(value) => onFieldChange('insurance_type', value)}
          />

          <div className="field">
            <label htmlFor="recent_bmi">Recent BMI (optional)</label>
            <input
              id="recent_bmi"
              type="number"
              min="0"
              max="100"
              step="0.1"
              value={fields.recent_bmi}
              onChange={(event) => onFieldChange('recent_bmi', event.target.value)}
            />
          </div>
        </div>

        <button type="submit" className="button primary" disabled={isPredicting}>
          {isPredicting ? 'Predicting…' : 'Predict'}
        </button>
      </form>
    </section>
  )
}
