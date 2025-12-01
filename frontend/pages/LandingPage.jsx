import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Brain, Smartphone, Zap, Shield, Eye, Target, Activity, Bot } from 'lucide-react'

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center max-w-4xl mx-auto">
          <div className="flex items-center justify-center gap-4 mb-6">
            <Brain className="w-16 h-16 text-primary animate-pulse" />
            <h1 className="text-6xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              AutoRL AI Agents
            </h1>
          </div>
          <p className="text-2xl text-muted-foreground mb-4">
            🤖 Autonomous AI Agents for Mobile Automation
          </p>
          <p className="text-xl font-semibold text-primary mb-8">
            Intelligent • Visual • Self-Learning
          </p>
          <p className="text-lg text-muted-foreground mb-12">
            Harness the power of <strong>AI Agents</strong> with LLM-powered task planning, 
            <strong> Visual Perception</strong> (OCR & UI detection), and <strong>Reinforcement Learning</strong> 
            to automate complex mobile workflows with natural language commands.
          </p>
          <div className="flex gap-4 justify-center">
            <Link to="/app/dashboard">
              <Button size="lg" className="text-lg px-8">
                Get Started
              </Button>
            </Link>
            <Link to="/app/docs">
              <Button size="lg" variant="outline" className="text-lg px-8">
                Documentation
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* AI Agent Features */}
      <div className="container mx-auto px-4 py-16">
        <h2 className="text-3xl font-bold text-center mb-4">
          <Bot className="inline w-8 h-8 mr-2 text-primary" />
          AI Agent Capabilities
        </h2>
        <p className="text-center text-muted-foreground mb-12 max-w-2xl mx-auto">
          Our AI agents go beyond simple chatbots - they see, think, act, and learn
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <Card className="border-primary/20 bg-gradient-to-br from-primary/10 to-primary/5">
            <CardContent className="p-6 text-center">
              <Brain className="w-12 h-12 text-primary mx-auto mb-4 animate-pulse" />
              <h3 className="text-xl font-bold mb-2">LLM-Powered Brain</h3>
              <p className="text-muted-foreground text-sm">
                Advanced language models for intelligent task planning and decision making
              </p>
            </CardContent>
          </Card>
          <Card className="border-blue-500/20 bg-gradient-to-br from-blue-500/10 to-blue-500/5">
            <CardContent className="p-6 text-center">
              <Eye className="w-12 h-12 text-blue-500 mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Visual Perception</h3>
              <p className="text-muted-foreground text-sm">
                OCR, screenshot analysis, and UI element detection for mobile understanding
              </p>
            </CardContent>
          </Card>
          <Card className="border-purple-500/20 bg-gradient-to-br from-purple-500/10 to-purple-500/5">
            <CardContent className="p-6 text-center">
              <Target className="w-12 h-12 text-purple-500 mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">RL Training</h3>
              <p className="text-muted-foreground text-sm">
                Reinforcement learning for continuous improvement from every execution
              </p>
            </CardContent>
          </Card>
          <Card className="border-orange-500/20 bg-gradient-to-br from-orange-500/10 to-orange-500/5">
            <CardContent className="p-6 text-center">
              <Activity className="w-12 h-12 text-orange-500 mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Autonomous Actions</h3>
              <p className="text-muted-foreground text-sm">
                Execute real actions: tap, swipe, type - not just text responses
              </p>
            </CardContent>
          </Card>
        </div>

        <h2 className="text-3xl font-bold text-center mb-12">Platform Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <CardContent className="p-6 text-center">
              <Smartphone className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Cross-Platform</h3>
              <p className="text-muted-foreground">
                Supports both Android and iOS automation
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <Zap className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Real-Time</h3>
              <p className="text-muted-foreground">
                Live execution monitoring with WebSocket updates
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <Shield className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Secure</h3>
              <p className="text-muted-foreground">
                PII masking and safety guardrails built-in
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <Bot className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Multi-Agent</h3>
              <p className="text-muted-foreground">
                Specialized agents for perception, planning, and execution
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
