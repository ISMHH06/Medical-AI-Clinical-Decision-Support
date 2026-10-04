import { useState } from 'react'
import './App.css'
import { explain, predict } from './api.js'
import ExplanationResults from './components/ExplanationResults.jsx'
import PredictionResults from './components/PredictionResults.jsx'
import UploadForm from './components/UploadForm.jsx'

const INITIAL_FIELDS = {
  age: '',
  sex: 'Female',
  race: 'White',
  ethnicity: 'Non-Hispanic/Non-Latino',
  insurance_type: 'Private Insurance',
  recent_bmi: '',
}

function ErrorBox({ message }) {
  if (!message) return null
  return (
    <div className="error-box" role="alert">
      <strong>Error:</strong> {message}
    </div>
  )
}

export default function App() {
  const [imageFile, setImageFile] = useState(null)
  const [fields, setFields] = useState(INITIAL_FIELDS)

  const [predictions, setPredictions] = useState(null)
  const [explanations, setExplanations] = useState(null)

  const [isPredicting, setIsPredicting] = useState(false)
  const [isExplaining, setIsExplaining] = useState(false)

  const [predictError, setPredictError] = useState(null)
  const [explainError, setExplainError] = useState(null)

  const updateField = (name, value) => {
    setFields((previous) => ({ ...previous, [name]: value }))
  }

  const handlePredict = async () => {
    setPredictError(null)
    setExplainError(null)

    if (!imageFile) {
      setPredictError('Please select a chest X-ray image first.')
      return
    }
    if (String(fields.age).trim() === '') {
      setPredictError('Age is required.')
      return
    }

    setIsPredicting(true)
    try {
      const result = await predict(imageFile, fields)
      setPredictions(result)
      setExplanations(null)
    } catch (error) {
      setPredictions(null)
      setExplanations(null)
      setPredictError(error.message)
    } finally {
      setIsPredicting(false)
    }
  }

  const handleExplain = async () => {
    setExplainError(null)

    if (!imageFile) {
      setExplainError('Please select a chest X-ray image first.')
      return
    }

    setIsExplaining(true)
    try {
      const result = await explain(imageFile, fields)
      setExplanations(result)
    } catch (error) {
      setExplanations(null)
      setExplainError(error.message)
    } finally {
      setIsExplaining(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Medical AI Clinical Decision Support</h1>
        <p className="app-subtitle">
          Multimodal chest X-ray prediction (vision + clinical). Research prototype — not a
          medical device.
        </p>
      </header>

      <ErrorBox message={predictError} />
      <ErrorBox message={explainError} />

      <UploadForm
        imageFile={imageFile}
        onImageChange={setImageFile}
        fields={fields}
        onFieldChange={updateField}
        onPredict={handlePredict}
        isPredicting={isPredicting}
      />

      <PredictionResults
        predictions={predictions}
        onExplain={handleExplain}
        isExplaining={isExplaining}
      />

      <ExplanationResults explanations={explanations} />
    </div>
  )
}
