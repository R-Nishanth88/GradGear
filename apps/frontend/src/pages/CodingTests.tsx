import React, { useEffect, useState } from 'react'
import Card from '../components/ui/Card'
import { getCodingTests } from '../lib/api'

export default function CodingTests() {
  const [items, setItems] = useState<any[]>([])
  useEffect(() => { getCodingTests().then(r => setItems(r.data.items || [])) }, [])
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-4">
      <Card>
        <div className="font-display text-xl mb-2">Coding Challenges</div>
        {items.map((c:any) => (
          <div key={c.id} className="mb-4">
            <div className="font-medium">{c.title}</div>
            <div className="text-sm text-slate-600 mb-2">{c.prompt}</div>
            <pre className="bg-slate-50 dark:bg-slate-800 p-3 rounded">{c.signature}</pre>
          </div>
        ))}
      </Card>
    </div>
  )
}


