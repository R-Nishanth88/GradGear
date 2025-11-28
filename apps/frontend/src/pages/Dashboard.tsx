import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import Sidebar from '../components/layout/Sidebar'
import Card from '../components/ui/Card'
import { ResponsiveContainer, RadialBarChart, RadialBar, BarChart, Bar, XAxis, Tooltip } from 'recharts'
import { getUserDomain } from '../lib/api'
import authStore from '../store/auth'

const skillScore = [{ name: 'Score', value: 78, fill: '#3B82F6' }]
const trending = [
  { name: 'GenAI', val: 40 },
  { name: 'K8s', val: 27 },
  { name: 'Spark', val: 22 },
]

export default function Dashboard() {
  const [domain, setDomain] = useState<string>('')
  const user = authStore(s => s.user)
  
  useEffect(() => {
    getUserDomain().then(r => setDomain(r.data.domains?.[0] || '')).catch(() => {})
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="flex min-h-screen"
    >
      <Sidebar />
      <div className="flex-1 p-6 space-y-6">
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mb-6"
        >
          <h1 className="font-display text-3xl mb-2">Welcome back, {user?.name || 'User'}!</h1>
          <p className="text-slate-600 dark:text-slate-400">Here's your personalized career journey overview.</p>
        </motion.div>

        {domain && (
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="inline-block px-4 py-2 rounded-full bg-gradient-to-r from-brand-blue/10 to-brand-purple/10 border border-brand-blue/30"
          >
            <span className="text-sm">
              Your Focus Domain:{' '}
              <span className="font-semibold text-brand-blue">{domain}</span>
            </span>
          </motion.div>
        )}

        <div className="grid md:grid-cols-3 gap-6">
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <Card>
              <div className="font-medium mb-2">Your Skill Score</div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart innerRadius="60%" outerRadius="100%" data={skillScore} startAngle={90} endAngle={-270}>
                    <RadialBar dataKey="value" cornerRadius={10} />
                  </RadialBarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </motion.div>
          <motion.div
            className="md:col-span-2"
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <Card>
              <div className="font-medium mb-2">Trending Market Skills</div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={trending}>
                    <XAxis dataKey="name" />
                    <Tooltip />
                    <Bar dataKey="val" fill="#9333EA" radius={[6,6,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </motion.div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <Card>
              <div className="font-medium mb-2">Recommended Projects</div>
              <ul className="space-y-2 text-sm">
                <li className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50">Spark Job Market Analyzer — <span className="text-xs">Hard</span></li>
                <li className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50">Resume Parser with SpaCy — <span className="text-xs">Medium</span></li>
              </ul>
            </Card>
          </motion.div>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.7 }}
          >
            <Card>
              <div className="font-medium mb-2">Learning Path Progress</div>
              <ol className="relative border-s border-slate-200 dark:border-slate-700 ml-3 text-sm">
                {['Foundations','Projects','Interview Prep'].map((s, i) => (
                  <li key={s} className="mb-4 ms-4">
                    <div className="absolute w-2 h-2 bg-brand-blue rounded-full -start-1.5 mt-1" />
                    <time className="text-xs text-slate-500">Step {i+1}</time>
                    <div className="font-medium">{s}</div>
                  </li>
                ))}
              </ol>
            </Card>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
