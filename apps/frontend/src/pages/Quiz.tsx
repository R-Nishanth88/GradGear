import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import { getUserDomain, getQuiz, trackProgress } from '../lib/api'
import { getProfile } from '../lib/api'

type Question = {
  q: string
  options: string[]
  a: number  // index of correct answer
}

export default function Quiz() {
  const [domain, setDomain] = useState('AI/ML')
  const [items, setItems] = useState<Question[]>([])
  const [selectedAnswers, setSelectedAnswers] = useState<{ [key: number]: number }>({})
  const [submitted, setSubmitted] = useState(false)
  const [score, setScore] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadQuiz()
  }, [])

  async function loadQuiz() {
    setLoading(true)
    setError(null)
    
    let userDomain = 'AI/ML'
    
    // Try multiple methods to get user domain
    try {
      // Method 1: Try getUserDomain endpoint
      try {
        const domainRes = await getUserDomain()
        if (domainRes.data?.domains?.[0]) {
          userDomain = domainRes.data.domains[0]
        }
      } catch (e) {
        console.warn('getUserDomain failed, trying profile:', e)
        // Method 2: Try getProfile to get domain
        try {
          const profile = await getProfile()
          if (profile.data?.domain) {
            userDomain = profile.data.domain
          }
        } catch (e2) {
          console.warn('getProfile failed, using default domain:', e2)
        }
      }
    } catch (e) {
      console.warn('Domain detection failed, using default:', e)
    }

    setDomain(userDomain)

    // Try to load quiz with detected domain
    try {
      const res = await getQuiz(userDomain)
      if (res.data?.items && res.data.items.length > 0) {
        setItems(res.data.items)
        setError(null)
      } else {
        throw new Error('No quiz questions available')
      }
    } catch (e: any) {
      console.error('Quiz load error:', e)
      const status = e?.response?.status
      const errorMsg = e?.response?.data?.detail || e?.message || 'Failed to load quiz'
      
      // If 401, it's an auth issue
      if (status === 401) {
        setError('Please log in to access quizzes')
        return
      }
      
      // If domain-specific quiz fails, try default 'AI/ML'
      if (userDomain !== 'AI/ML') {
        setError(`Trying default domain...`)
        try {
          const res = await getQuiz('AI/ML')
          if (res.data?.items && res.data.items.length > 0) {
            setItems(res.data.items)
            setDomain('AI/ML')
            setError(null)
            return
          }
        } catch (e2: any) {
          console.error('Fallback quiz load failed:', e2)
          setError(`Failed to load quiz: ${errorMsg}. Please check your connection.`)
        }
      } else {
        setError(`Failed to load quiz: ${errorMsg}`)
      }
    } finally {
      setLoading(false)
    }
  }

  function handleAnswerSelect(questionIdx: number, optionIdx: number) {
    if (submitted) return
    setSelectedAnswers({ ...selectedAnswers, [questionIdx]: optionIdx })
  }

  async function handleSubmit() {
    if (Object.keys(selectedAnswers).length < items.length) {
      toast.error('Please answer all questions')
      return
    }

    // Calculate score
    let correct = 0
    items.forEach((q, idx) => {
      if (selectedAnswers[idx] === q.a) {
        correct++
      }
    })
    const scorePercent = Math.round((correct / items.length) * 100)
    setScore(scorePercent)
    setSubmitted(true)

    // Track progress
    try {
      await trackProgress('quiz_passed', {
        domain,
        score: scorePercent,
        total_questions: items.length,
        correct_answers: correct,
      })
      toast.success(`Quiz completed! Score: ${scorePercent}%`)
    } catch (e) {
      console.error('Failed to track progress:', e)
      toast.success(`Quiz completed! Score: ${scorePercent}%`)
    }
  }

  function handleReset() {
    setSelectedAnswers({})
    setSubmitted(false)
    setScore(null)
    loadQuiz()
  }

  const getAnswerColor = (questionIdx: number, optionIdx: number) => {
    if (!submitted) {
      return selectedAnswers[questionIdx] === optionIdx
        ? 'bg-brand-blue text-white border-brand-blue'
        : 'bg-slate-50 dark:bg-slate-800/60 border-slate-200 dark:border-slate-700'
    }

    const question = items[questionIdx]
    const isCorrect = optionIdx === question.a
    const isSelected = selectedAnswers[questionIdx] === optionIdx

    if (isCorrect) {
      return 'bg-green-100 dark:bg-green-900/30 border-green-500 text-green-900 dark:text-green-100'
    }
    if (isSelected && !isCorrect) {
      return 'bg-red-100 dark:bg-red-900/30 border-red-500 text-red-900 dark:text-red-100'
    }
    return 'bg-slate-50 dark:bg-slate-800/60 border-slate-200 dark:border-slate-700'
  }

  return (
    <div className="min-h-screen gradient-bg p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card>
            <div className="flex justify-between items-center mb-6">
              <div>
                <h1 className="font-display text-3xl mb-2">{domain} Quiz</h1>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Test your knowledge in {domain}
                </p>
              </div>
              {score !== null && (
                <div className="text-right">
                  <div className="text-4xl font-bold text-brand-blue">{score}%</div>
                  <div className="text-sm text-slate-600 dark:text-slate-400">Score</div>
                </div>
              )}
            </div>

            {loading && (
              <div className="flex items-center justify-center py-12">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-8 h-8 border-2 border-brand-blue border-t-transparent rounded-full"
                />
              </div>
            )}

            {error && !loading && (
              <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                <p className="text-red-900 dark:text-red-200">{error}</p>
                <Button onClick={loadQuiz} variant="ghost" className="mt-2 text-sm">
                  Try Again
                </Button>
              </div>
            )}

            {!loading && !error && items.length > 0 && (
              <>
                <div className="space-y-6 mb-6">
                  {items.map((q, questionIdx) => (
                    <motion.div
                      key={questionIdx}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: questionIdx * 0.1 }}
                      className="p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50"
                    >
                      <div className="font-medium mb-3 text-slate-900 dark:text-slate-100">
                        {questionIdx + 1}. {q.q}
                      </div>
                      <div className="grid md:grid-cols-2 gap-3">
                        {q.options.map((option, optionIdx) => (
                          <button
                            key={optionIdx}
                            onClick={() => handleAnswerSelect(questionIdx, optionIdx)}
                            disabled={submitted}
                            className={`p-3 rounded-lg border-2 transition text-left ${
                              submitted ? 'cursor-default' : 'cursor-pointer hover:scale-105'
                            } ${getAnswerColor(questionIdx, optionIdx)}`}
                          >
                            <div className="flex items-center gap-2">
                              <span className="font-medium">
                                {String.fromCharCode(65 + optionIdx)}.
                              </span>
                              <span>{option}</span>
                              {submitted && optionIdx === q.a && (
                                <span className="ml-auto text-green-600 dark:text-green-400">✓</span>
                              )}
                              {submitted && selectedAnswers[questionIdx] === optionIdx && optionIdx !== q.a && (
                                <span className="ml-auto text-red-600 dark:text-red-400">✗</span>
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    </motion.div>
                  ))}
                </div>

                <div className="flex gap-3">
                  {!submitted ? (
                    <Button
                      onClick={handleSubmit}
                      disabled={Object.keys(selectedAnswers).length < items.length}
                      className="flex-1"
                    >
                      Submit Quiz ({Object.keys(selectedAnswers).length}/{items.length} answered)
                    </Button>
                  ) : (
                    <>
                      <Button onClick={handleReset} variant="ghost" className="flex-1">
                        Try Again
                      </Button>
                      <Button onClick={loadQuiz} className="flex-1">
                        New Quiz
                      </Button>
                    </>
                  )}
                </div>

                {submitted && score !== null && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="mt-6 p-6 rounded-lg bg-gradient-to-r from-brand-blue/10 to-brand-purple/10 border border-brand-blue/20"
                  >
                    <div className="text-center">
                      <div className="text-5xl font-bold text-brand-blue mb-2">{score}%</div>
                      <div className="text-lg font-medium text-slate-700 dark:text-slate-300 mb-1">
                        {score >= 80 ? 'Excellent!' : score >= 60 ? 'Good Job!' : 'Keep Practicing!'}
                      </div>
                      <div className="text-sm text-slate-600 dark:text-slate-400">
                        You got {Object.values(selectedAnswers).filter((ans, idx) => ans === items[idx].a).length} out of {items.length} questions correct
                      </div>
                    </div>
                  </motion.div>
                )}
              </>
            )}
          </Card>
        </motion.div>
      </div>
    </div>
  )
}
