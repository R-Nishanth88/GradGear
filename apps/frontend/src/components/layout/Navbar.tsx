import React from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import Button from '../ui/Button'
import useAppStore from '../../store/useAppStore'
import authStore from '../../store/auth'

export default function Navbar() {
  const dark = useAppStore(s => s.darkMode)
  const toggle = useAppStore(s => s.toggleDark)
  const user = authStore(s => s.user)
  const logout = authStore(s => s.logout)
  const navigate = useNavigate()
  const location = useLocation()
  
  const isHomePage = location.pathname === '/' || location.pathname === '/dashboard'
  const textColor = isHomePage ? 'text-white' : 'text-slate-600 dark:text-slate-300'
  const hoverColor = isHomePage ? 'hover:text-white/80' : 'hover:text-slate-900 dark:hover:text-white'

  function handleLogout() {
    logout()
    navigate('/login')
  }
  
  return (
    <motion.div
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`sticky top-0 z-40 backdrop-blur ${
        isHomePage 
          ? 'supports-[backdrop-filter]:bg-black/20 border-b border-white/10' 
          : 'supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-slate-900/40 border-b'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-4">
        <Link to="/dashboard" className={`font-display text-xl flex items-center gap-2 ${textColor}`}>
          ⚙️ GradGear
        </Link>
        <div className={`hidden md:flex items-center gap-4 text-sm ${textColor}`}>
          <Link to="/dashboard" className={`${hoverColor} transition`}>Dashboard</Link>
          <Link to="/skill-dashboard" className={`${hoverColor} transition`}>Skills</Link>
          <Link to="/resume" className={`${hoverColor} transition`}>Resume</Link>
          <Link to="/learning" className={`${hoverColor} transition`}>Learning</Link>
          <Link to="/quiz" className={`${hoverColor} transition`}>Quiz</Link>
          <Link to="/coding" className={`${hoverColor} transition`}>Coding</Link>
          <Link to="/projects" className={`${hoverColor} transition`}>Projects</Link>
          <Link to="/leaderboard" className={`${hoverColor} transition`}>Leaderboard</Link>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={toggle}
            className={`p-2 rounded-lg ${isHomePage ? 'hover:bg-white/20 text-white' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}
          >
            {dark ? '☀️' : '🌙'}
          </motion.button>
          <div className={`flex items-center gap-2 text-sm ${textColor}`}>
            <span>👤 {user?.name}</span>
          </div>
          <Button 
            variant="ghost" 
            onClick={handleLogout}
            className={isHomePage ? 'text-white border-white/30 hover:bg-white/20' : ''}
          >
            Logout
          </Button>
        </div>
      </div>
    </motion.div>
  )
}