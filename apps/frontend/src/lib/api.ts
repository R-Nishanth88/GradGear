import axios from 'axios'
import authStore from '../store/auth'

const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const api = axios.create({ baseURL: apiBase + '/api' })

api.interceptors.request.use((config) => {
  const token = authStore.getState().token
  if (token) {
    config.headers = config.headers || {}
    ;(config.headers as any)['Authorization'] = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err?.response?.status === 401) {
      authStore.getState().logout()
      if (typeof window !== 'undefined') window.location.href = '/login'
    }
    // Better error logging
    if (err?.response) {
      console.error('API Error:', err.response.status, err.response.data)
    }
    return Promise.reject(err)
  }
)

// Mock helpers
function mock<T>(data: T, delay = 600): Promise<{ data: T }> {
  return new Promise(resolve => setTimeout(() => resolve({ data }), delay))
}

export function analyzeSkills(payload: { skills?: string[]; resume?: string }) {
  // return api.post('/skills/analyze', payload)
  return mock({
    missingSkills: [
      { name: 'TensorFlow', level: 'Intermediate', trend: [12,14,18,20] },
      { name: 'Kubernetes', level: 'Beginner', trend: [5,7,9,12] },
    ],
    projects: [
      { title: 'ML Pipeline on Kubeflow', tags: ['ML','K8s'], difficulty: 'Hard', github: 'https://github.com/search?q=kubeflow' },
      { title: 'Resume Parser with SpaCy', tags: ['NLP'], difficulty: 'Medium', github: 'https://github.com/search?q=spacy+resume' }
    ],
    courses: [
      { title: 'DeepLearning.AI TensorFlow', provider: 'Coursera' },
      { title: 'Kubernetes for Devs', provider: 'Udemy' }
    ]
  })
}

export function recommendProjects(payload: { skills: string[] }) {
  // return api.post('/projects/recommend', payload)
  return mock({
    items: [
      { title: 'Real-time Skill Tracker', tags: ['React','Socket.io'], difficulty: 'Medium' },
      { title: 'Spark Job Market Analyzer', tags: ['Spark','MLlib'], difficulty: 'Hard' }
    ]
  })
}

export function generateResume(payload: { summary: string; skills: string[] }) {
  // return api.post('/resume/generate', payload)
  return mock({
    content: 'Generated resume content with quantified achievements and ATS-friendly keywords.'
  })
}

export function trendsSkills() {
  // return api.get('/trends/skills')
  return mock({
    labels: ['Jan','Feb','Mar','Apr','May','Jun'],
    series: [
      { name: 'GenAI', data: [10,14,20,28,34,40] },
      { name: 'Kubernetes', data: [8,11,16,19,23,27] }
    ]
  })
}

export default api

// Domain preferences
export function getUserDomain() { return api.get('/user/domain') }
export function setUserDomain(domains: string[]) { return api.post('/user/domain', { domains }) }

// Auth
export function registerUser(payload: {name:string;email:string;password:string;college?:string;year?:string;domain?:string}) {
  return api.post('/register', payload)
}
export function loginUser(payload: {email:string;password:string}) {
  return api.post('/login', payload)
}
export function getProfile() { return api.get('/user/profile') }

// Resume
export function uploadResume(file: File) {
  const form = new FormData()
  form.append('file', file)
  return api.post('/resume/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
}

export function analyzeResume(resumeId?: number) {
  if (resumeId) {
    const form = new FormData()
    form.append('resume_id', resumeId.toString())
    return api.post('/resume/analyze', form)
  } else {
    // Send empty JSON body when no resume_id (will use latest resume)
    return api.post('/resume/analyze', {})
  }
}

export function generateResumeApi(payload: {
  name: string
  domain: string
  skills: string[]
  achievements: string[]
  personal?: { [key: string]: string }
  education?: Array<{ [key: string]: string }>
  experience?: Array<{ [key: string]: any }>
  projects?: Array<{ [key: string]: string }>
}) {
  return api.post('/resume/generate', payload)
}

export function getSkillRecommendations(domain: string, skills?: string[]) {
  const params = new URLSearchParams()
  params.append('domain', domain)
  if (skills) params.append('skills', skills.join(','))
  return api.get(`/resume/recommendations/skills?${params.toString()}`)
}

export function trackProgress(activityType: string, activityData: any) {
  const form = new FormData()
  form.append('activity_type', activityType)
  form.append('activity_data', JSON.stringify(activityData))
  return api.post('/resume/track-progress', form)
}

// Recommendations
export function getRecommendations(domain: string) { return api.get(`/recommendations/${encodeURIComponent(domain)}`) }

// Quiz & Coding
export function getQuiz(domain: string) { return api.get(`/quiz/${encodeURIComponent(domain)}`) }
export function getCodingTests() { return api.get('/codingtest') }

// Leaderboard
export function getLeaderboard(limit?: number) { return api.get('/leaderboard', { params: { limit } }) }

// Progress
export function getUserProgress() { return api.get('/user/progress') }


