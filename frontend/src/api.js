const API_BASE = 'http://localhost:8000'

function buildFormData(imageFile, clinicalFields) {
  const formData = new FormData()
  formData.append('image', imageFile)
  formData.append('age', String(clinicalFields.age))
  formData.append('sex', clinicalFields.sex)
  formData.append('race', clinicalFields.race)
  formData.append('ethnicity', clinicalFields.ethnicity)
  formData.append('insurance_type', clinicalFields.insurance_type)
  const bmi = clinicalFields.recent_bmi
  if (bmi !== null && bmi !== undefined && String(bmi).trim() !== '') {
    formData.append('recent_bmi', String(bmi))
  }
  return formData
}

function extractDetail(body, status) {
  if (body && typeof body.detail === 'string') return body.detail
  if (body && body.detail !== undefined) return JSON.stringify(body.detail)
  return `Request failed with status ${status}`
}

async function postForm(path, imageFile, clinicalFields) {
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      body: buildFormData(imageFile, clinicalFields),
    })
  } catch (networkError) {
    throw new Error(
      `Could not reach the API at ${API_BASE}. Is the backend running? (${networkError.message})`,
    )
  }

  let body = null
  try {
    body = await response.json()
  } catch {
    // Non-JSON response; fall through to the status-based error below.
  }

  if (!response.ok) {
    throw new Error(extractDetail(body, response.status))
  }
  return body
}

export async function predict(imageFile, clinicalFields) {
  const body = await postForm('/predict', imageFile, clinicalFields)
  return body?.predictions ?? null
}

export async function explain(imageFile, clinicalFields) {
  const body = await postForm('/explain', imageFile, clinicalFields)
  return body?.explanations ?? null
}
