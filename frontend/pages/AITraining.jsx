import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Brain, Play, Pause, TrendingUp } from 'lucide-react'

export default function AITraining() {
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress] = useState(65)
  
  const models = [
    { name: 'v1.0', accuracy: 87.3, episodes: 1000, status: 'deprecated' },
    { name: 'v2.1', accuracy: 92.1, episodes: 2500, status: 'active' },
    { name: 'v3.0', accuracy: 94.8, episodes: 5000, status: 'latest' }
  ]

  return (
    <div className="container mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">AI Training Center</h1>
        <p className="text-muted-foreground">
          Manage reinforcement learning models and training
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Brain className="w-5 h-5 mr-2 text-primary" />
              Training Progress
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Current Episode: 3250 / 5000</span>
                <span className="text-sm text-muted-foreground">{trainingProgress}%</span>
              </div>
              <Progress value={trainingProgress} />
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Reward</p>
                <p className="font-bold text-lg">+127.3</p>
              </div>
              <div>
                <p className="text-muted-foreground">Loss</p>
                <p className="font-bold text-lg">0.023</p>
              </div>
            </div>
            <Button
              className="w-full"
              onClick={() => setIsTraining(!isTraining)}
            >
              {isTraining ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
              {isTraining ? 'Pause Training' : 'Start Training'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="w-5 h-5 mr-2 text-primary" />
              Training Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-3 border rounded-lg">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-muted-foreground">Success Rate</span>
                  <span className="font-bold">94.2%</span>
                </div>
                <Progress value={94.2} />
              </div>
              <div className="p-3 border rounded-lg">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-muted-foreground">Avg Reward</span>
                  <span className="font-bold">+89.5</span>
                </div>
                <Progress value={89.5} />
              </div>
              <div className="p-3 border rounded-lg">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-muted-foreground">Model Accuracy</span>
                  <span className="font-bold">96.1%</span>
                </div>
                <Progress value={96.1} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Model Versions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {models.map((model) => (
              <div
                key={model.name}
                className="p-4 border rounded-lg flex items-center justify-between hover:bg-accent/50 transition-colors"
              >
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <span className="font-bold">{model.name}</span>
                    <Badge variant={model.status === 'latest' ? 'default' : 'secondary'}>
                      {model.status}
                    </Badge>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {model.episodes} episodes • {model.accuracy}% accuracy
                  </div>
                </div>
                <Button variant="outline" size="sm">
                  Load Model
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
