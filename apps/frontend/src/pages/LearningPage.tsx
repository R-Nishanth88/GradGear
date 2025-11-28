import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { useDomain, Domain, Role, Level, Duration } from '../contexts/DomainProvider'
import { enhancedDomainApi } from '../lib/enhancedApi'
import LearningHeader from '../components/learning/LearningHeader'
import PersonalizedStart from '../components/learning/PersonalizedStart'
import SkillTracks from '../components/learning/SkillTracks'
import ResourcesHub from '../components/learning/ResourcesHub'
import CertificationRoadmap from '../components/learning/CertificationRoadmap'
import BeginnersChecklist from '../components/learning/BeginnersChecklist'
import PracticeAssess from '../components/learning/PracticeAssess'
import PlaylistHistory from '../components/learning/PlaylistHistory'
import LearningFooter from '../components/learning/LearningFooter'
import ErrorBoundary from '../components/learning/ErrorBoundary'
import LoadingSpinner from '../components/ui/LoadingSpinner'

const LearningPage: React.FC = () => {
  const { domain, filters, setDomain } = useDomain()
  const queryClient = useQueryClient()
  const [activeSection, setActiveSection] = useState<string>('start')

  // Domain change handler - invalidate all queries
  const handleDomainChange = (newDomain: Domain) => {
    setDomain(newDomain)
    // Invalidate all learning-related queries
    queryClient.invalidateQueries({ queryKey: ['recommendations'] })
    queryClient.invalidateQueries({ queryKey: ['trendingSkills'] })
    queryClient.invalidateQueries({ queryKey: ['resources'] })
    queryClient.invalidateQueries({ queryKey: ['certifications'] })
    queryClient.invalidateQueries({ queryKey: ['quiz'] })
    queryClient.invalidateQueries({ queryKey: ['codingTest'] })
    
    toast.success(`Switched to ${newDomain.replace('_', '/').toUpperCase()} domain`)
  }

  // Update API domain when context domain changes
  useEffect(() => {
    enhancedDomainApi.setDomain(domain)
  }, [domain])

  // Page fade-in animation
  const pageVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.6,
        staggerChildren: 0.1
      }
    }
  }

  const sectionVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.4 }
    }
  }

  // Domain accent color mapping
  const getDomainAccent = (currentDomain: Domain) => {
    const accents = {
      cybersec: 'text-domain-cybersec border-domain-cybersec',
      ai_ml: 'text-domain-ai_ml border-domain-ai_ml',
      data: 'text-domain-data border-domain-data',
      web: 'text-domain-web border-domain-web',
      cloud: 'text-domain-cloud border-domain-cloud',
      iot: 'text-domain-iot border-domain-iot'
    }
    return accents[currentDomain]
  }

  // Domain background color mapping
  const getDomainBg = (currentDomain: Domain) => {
    const bgColors = {
      cybersec: 'bg-domain-cybersec/10 hover:bg-domain-cybersec/20',
      ai_ml: 'bg-domain-ai_ml/10 hover:bg-domain-ai_ml/20',
      data: 'bg-domain-data/10 hover:bg-domain-data/20',
      web: 'bg-domain-web/10 hover:bg-domain-web/20',
      cloud: 'bg-domain-cloud/10 hover:bg-domain-cloud/20',
      iot: 'bg-domain-iot/10 hover:bg-domain-iot/20'
    }
    return bgColors[currentDomain]
  }

  if (!domain) {
    return (
      <div className="min-h-screen bg-dark-bg-primary flex items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  return (
    <motion.div 
      className="min-h-screen bg-dark-bg-primary text-dark-text-primary"
      initial="hidden"
      animate="visible"
      variants={pageVariants}
    >
      {/* Header */}
      <motion.div variants={sectionVariants}>
        <LearningHeader 
          domain={domain}
          onDomainChange={handleDomainChange}
          filters={filters}
          activeSection={activeSection}
          onSectionChange={setActiveSection}
        />
      </motion.div>

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <AnimatePresence mode="wait">
          {/* Personalized Start Section */}
          {activeSection === 'start' && (
            <motion.div
              key="start"
              variants={sectionVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="mb-12"
            >
              <ErrorBoundary>
                <PersonalizedStart 
                  domain={domain}
                  filters={filters}
                  accentClass={getDomainAccent(domain)}
                  bgClass={getDomainBg(domain)}
                />
              </ErrorBoundary>
            </motion.div>
          )}

          {/* Skill Tracks Section */}
          {activeSection === 'skills' && (
            <motion.div
              key="skills"
              variants={sectionVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="mb-12"
            >
              <ErrorBoundary>
                <SkillTracks 
                  domain={domain}
                  accentClass={getDomainAccent(domain)}
                  bgClass={getDomainBg(domain)}
                />
              </ErrorBoundary>
            </motion.div>
          )}

          {/* Resources Hub Section */}
          {activeSection === 'resources' && (
            <motion.div
              key="resources"
              variants={sectionVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="mb-12"
            >
              <ErrorBoundary>
                <ResourcesHub 
                  domain={domain}
                  filters={filters}
                  accentClass={getDomainAccent(domain)}
                  bgClass={getDomainBg(domain)}
                />
              </ErrorBoundary>
            </motion.div>
          )}

          {/* Certification Roadmap Section */}
          {activeSection === 'certifications' && (
            <motion.div
              key="certifications"
              variants={sectionVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="mb-12"
            >
              <ErrorBoundary>
                <CertificationRoadmap 
                  domain={domain}
                  accentClass={getDomainAccent(domain)}
                  bgClass={getDomainBg(domain)}
                />
              </ErrorBoundary>
            </motion.div>
          )}

          {/* Beginners Checklist Section */}
          {activeSection === 'beginners' && (
            <motion.div
              key="beginners"
              variants={sectionVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="mb-12"
            >
              <ErrorBoundary>
                <BeginnersChecklist 
                  domain={domain}
                  accentClass={getDomainAccent(domain)}
                  bgClass={getDomainBg(domain)}
                />
              </ErrorBoundary>
            </motion.div>
          )}

          {/* Practice & Assess Section */}
          {activeSection === 'practice' && (
            <motion.div
              key="practice"
              variants={sectionVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="mb-12"
            >
              <ErrorBoundary>
                <PracticeAssess 
                  domain={domain}
                  accentClass={getDomainAccent(domain)}
                  bgClass={getDomainBg(domain)}
                />
              </ErrorBoundary>
            </motion.div>
          )}

          {/* Playlist & History Section */}
          {activeSection === 'playlist' && (
            <motion.div
              key="playlist"
              variants={sectionVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="mb-12"
            >
              <ErrorBoundary>
                <PlaylistHistory 
                  domain={domain}
                  accentClass={getDomainAccent(domain)}
                  bgClass={getDomainBg(domain)}
                />
              </ErrorBoundary>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <motion.div variants={sectionVariants}>
        <LearningFooter 
          domain={domain}
          accentClass={getDomainAccent(domain)}
        />
      </motion.div>
    </motion.div>
  )
}

export default LearningPage