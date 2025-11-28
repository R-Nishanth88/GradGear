import React, { useState } from 'react'
import Button from '../components/ui/Button'
import Card from '../components/ui/Card'
import { analyzeSkills } from '../lib/api'

export default function Skills() {
  const [skills, setSkills] = useState('Python, React, SQL')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    const res = await analyzeSkills({ skills: skills.split(',').map(s => s.trim()) })
    setResult(res.data)
    setLoading(false)
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <Card>
        <form onSubmit={onSubmit} className="flex flex-col md:flex-row gap-3 items-center">
          <input value={skills} onChange={e=>setSkills(e.target.value)} placeholder="Add your skills or upload resume" className="flex-1 px-3 py-2 rounded-lg border" />
          <Button type="submit">Analyze</Button>
        </form>
      </Card>

      {loading && <Card>Analyzing…</Card>}
      {result && (
        <div className="grid md:grid-cols-3 gap-4">
          <Card>
            <div className="font-medium mb-2">Missing Skills</div>
            <div className="flex flex-wrap gap-2 text-sm">
              {result.missingSkills.map((m:any) => (
                <span key={m.name} className="px-2 py-1 rounded-full bg-slate-100 dark:bg-slate-800">{m.name} · {m.level}</span>
              ))}
            </div>
          </Card>
          <Card>
            <div className="font-medium mb-2">Recommended Projects</div>
            <ul className="space-y-2 text-sm">
              {result.projects.map((p:any) => (
                <li key={p.title} className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/50">{p.title} — {p.difficulty}</li>
              ))}
            </ul>
          </Card>
          <Card>
            <div className="font-medium mb-2">Suggested Courses</div>
            <ul className="space-y-2 text-sm">
              {result.courses.map((c:any) => (
                <li key={c.title}>{c.title} · {c.provider}</li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}


