import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Play, Pause, Square, Edit, CheckCircle2, Clock, 
  AlertCircle, Brain, Zap, Eye, ArrowRight, 
  Sparkles, Target, TrendingUp, Info
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { apiService } from '@/services/api'
import { useWebSocket } from '@/hooks/useWebSocket'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { PlanFlowDiagram } from '@/components/PlanFlowDiagram'

export function CloudPlannerView({ instruction, onPlanGenerated, onExecute }) {
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [isExecuting, setIsExecuting] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [selectedAlternative, setSelectedAlternative] = useState(null)
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [editedPlan, setEditedPlan] = useState(null)

  // Generate plan when instruction changes
  useEffect(() => {
    if (instruction && instruction.trim()) {
      generatePlan(instruction)
    }
  }, [instruction])

  const generatePlan = async (taskInstruction) => {
    setLoading(true)
    setError(null)
    try {
      const response = await apiService.createDetailedPlan(taskInstruction)
      if (response.plan) {
        setPlan(response.plan)
        setCurrentStep(0)
        if (onPlanGenerated) {
          onPlanGenerated(response.plan)
        }
      }
    } catch (err) {
      console.error('Error generating plan:', err)
      setError(err.message || 'Failed to generate plan')
    } finally {
      setLoading(false)
    }
  }

  // Handle WebSocket updates for plan execution
  const handleWebSocketMessage = useCallback((message) => {
    if (!plan) return

    // Update current step based on execution events
    if (message.event === 'perception') {
      setCurrentStep(0)
    } else if (message.event === 'planning') {
      setCurrentStep(0)
    } else if (message.event === 'execution_start') {
      // If step number is provided, use it; otherwise increment
      if (message.step !== undefined) {
        setCurrentStep(Math.min(message.step, plan.steps.length))
      } else {
        setCurrentStep(prev => Math.min(prev + 1, plan.steps.length))
      }
    } else if (message.event === 'step_completed') {
      // Explicit step completion event
      if (message.step !== undefined) {
        setCurrentStep(Math.min(message.step + 1, plan.steps.length))
      } else {
        setCurrentStep(prev => Math.min(prev + 1, plan.steps.length))
      }
    } else if (message.event === 'task_completed') {
      setCurrentStep(plan.steps.length)
      setIsExecuting(false)
      setIsPaused(false)
    } else if (message.event === 'task_failed') {
      setIsExecuting(false)
      setIsPaused(false)
    }
  }, [plan])

  const { isConnected } = useWebSocket(handleWebSocketMessage)

  const handleExecute = () => {
    if (!plan) return
    setIsExecuting(true)
    setIsPaused(false)
    setCurrentStep(0)
    if (onExecute) {
      onExecute(plan)
    }
  }

  const handlePause = () => {
    setIsPaused(true)
  }

  const handleResume = () => {
    setIsPaused(false)
  }

  const handleCancel = () => {
    setIsExecuting(false)
    setIsPaused(false)
    setCurrentStep(0)
  }

  const handleSelectAlternative = (altIndex) => {
    if (plan && plan.alternatives && plan.alternatives[altIndex]) {
      const alt = plan.alternatives[altIndex]
      setSelectedAlternative(altIndex)
      // Create a new plan based on alternative
      const newPlan = {
        ...plan,
        steps: alt.steps.map((step, idx) => ({
          ...step,
          step_number: idx + 1,
          id: `${plan.plan_id}_alt_${altIndex}_step_${idx + 1}`
        })),
        total_steps: alt.steps.length,
        overall_confidence: alt.confidence,
        reasoning: `${plan.reasoning} (Alternative: ${alt.approach})`
      }
      setPlan(newPlan)
      setCurrentStep(0)
    }
  }

  const handleEditPlan = () => {
    setEditedPlan(JSON.parse(JSON.stringify(plan)))
    setShowEditDialog(true)
  }

  const handleSaveEdit = () => {
    if (editedPlan) {
      setPlan(editedPlan)
      setShowEditDialog(false)
    }
  }

  if (loading) {
    return (
      <Card className="border-primary/20">
        <CardContent className="p-8">
          <div className="flex flex-col items-center justify-center gap-4">
            <div className="relative">
              <Brain className="w-12 h-12 text-primary animate-pulse" />
              <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl animate-pulse" />
            </div>
            <div className="text-center">
              <p className="text-lg font-semibold">Generating Plan...</p>
              <p className="text-sm text-muted-foreground mt-1">
                AI is analyzing your task and creating an execution plan
              </p>
            </div>
            <Progress value={66} className="w-full max-w-md" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="border-destructive/20">
        <CardContent className="p-6">
          <div className="flex items-center gap-3 text-destructive">
            <AlertCircle className="w-5 h-5" />
            <div>
              <p className="font-semibold">Error generating plan</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!plan) {
    return (
      <Card className="border-primary/20">
        <CardContent className="p-8">
          <div className="text-center text-muted-foreground">
            <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>Enter a task instruction to generate a plan</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const progress = plan.steps.length > 0 ? (currentStep / plan.steps.length) * 100 : 0
  const activePlan = selectedAlternative !== null && plan.alternatives 
    ? plan.alternatives[selectedAlternative] 
    : null

  return (
    <div className="space-y-6">
      {/* Plan Header */}
      <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-transparent">
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <CardTitle className="flex items-center gap-2 mb-2">
                <div className="p-2 rounded-lg bg-primary/20">
                  <Brain className="w-5 h-5 text-primary" />
                </div>
                <span>Cloud Planner - Execution Plan</span>
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                {plan.instruction}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="bg-primary/10 border-primary/30">
                <Target className="w-3 h-3 mr-1" />
                {plan.total_steps} steps
              </Badge>
              <Badge variant="outline" className="bg-green-500/10 border-green-500/30 text-green-500">
                <TrendingUp className="w-3 h-3 mr-1" />
                {Math.round(plan.overall_confidence * 100)}% confidence
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Reasoning */}
          <div className="mb-4 p-4 rounded-lg bg-muted/50 border border-border/50">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-xs font-medium text-muted-foreground mb-1">AI Reasoning</p>
                <p className="text-sm text-foreground">{plan.reasoning}</p>
              </div>
            </div>
          </div>

          {/* Execution Controls */}
          <div className="flex items-center gap-2 flex-wrap">
            {!isExecuting ? (
              <Button onClick={handleExecute} className="gap-2" size="lg">
                <Play className="w-4 h-4" />
                Execute Plan
              </Button>
            ) : (
              <>
                {isPaused ? (
                  <Button onClick={handleResume} variant="outline" className="gap-2">
                    <Play className="w-4 h-4" />
                    Resume
                  </Button>
                ) : (
                  <Button onClick={handlePause} variant="outline" className="gap-2">
                    <Pause className="w-4 h-4" />
                    Pause
                  </Button>
                )}
                <Button onClick={handleCancel} variant="destructive" className="gap-2">
                  <Square className="w-4 h-4" />
                  Cancel
                </Button>
              </>
            )}
            <Button onClick={handleEditPlan} variant="outline" className="gap-2">
              <Edit className="w-4 h-4" />
              Edit Plan
            </Button>
            <Button onClick={() => generatePlan(instruction)} variant="ghost" className="gap-2">
              <Sparkles className="w-4 h-4" />
              Regenerate
            </Button>
          </div>

          {/* Progress Bar */}
          {isExecuting && (
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Execution Progress</span>
                <span className="font-semibold text-foreground">
                  {currentStep} / {plan.steps.length} steps
                </span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Plan Steps Visualization - Two Views */}
      <Tabs defaultValue="flow" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="flow">Flow Diagram</TabsTrigger>
          <TabsTrigger value="list">Step List</TabsTrigger>
        </TabsList>
        <TabsContent value="flow" className="mt-4">
          <PlanFlowDiagram 
            plan={plan} 
            currentStep={currentStep} 
            isExecuting={isExecuting}
            isPaused={isPaused}
          />
        </TabsContent>
        <TabsContent value="list" className="mt-4">
          <Card className="border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                <span>Execution Steps</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {plan.steps.map((step, index) => {
                  const isActive = currentStep === index + 1
                  const isCompleted = currentStep > index + 1
                  const isPending = currentStep < index + 1

                  return (
                    <div
                      key={step.id || `step_${index}`}
                      className={cn(
                        "relative flex items-start gap-4 p-4 rounded-lg border transition-all duration-300",
                        isActive && "bg-primary/10 border-primary/50 shadow-lg shadow-primary/10",
                        isCompleted && "bg-green-500/5 border-green-500/30",
                        isPending && "bg-muted/30 border-border/50 opacity-60"
                      )}
                    >
                      {/* Step Number */}
                      <div className={cn(
                        "flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all",
                        isActive && "bg-primary text-primary-foreground scale-110",
                        isCompleted && "bg-green-500 text-white",
                        isPending && "bg-muted text-muted-foreground"
                      )}>
                        {isCompleted ? (
                          <CheckCircle2 className="w-5 h-5" />
                        ) : (
                          index + 1
                        )}
                      </div>

                      {/* Step Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-2 mb-1">
                          <p className={cn(
                            "font-semibold text-sm",
                            isActive && "text-primary",
                            isCompleted && "text-green-500",
                            isPending && "text-muted-foreground"
                          )}>
                            {step.description || step.action}
                          </p>
                          <Badge 
                            variant="outline" 
                            className={cn(
                              "text-xs",
                              step.confidence >= 0.9 && "border-green-500/30 bg-green-500/10",
                              step.confidence >= 0.8 && step.confidence < 0.9 && "border-yellow-500/30 bg-yellow-500/10",
                              step.confidence < 0.8 && "border-orange-500/30 bg-orange-500/10"
                            )}
                          >
                            {Math.round(step.confidence * 100)}%
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Badge variant="secondary" className="text-xs font-mono">
                            {step.action}
                          </Badge>
                          {step.target_id && (
                            <span className="text-muted-foreground">
                              → {step.target_id}
                            </span>
                          )}
                          {step.value && (
                            <span className="text-muted-foreground">
                              : "{step.value}"
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Active Indicator */}
                      {isActive && (
                        <div className="absolute right-4 top-4">
                          <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
                        </div>
                      )}

                      {/* Connector Line */}
                      {index < plan.steps.length - 1 && (
                        <div className={cn(
                          "absolute left-5 top-14 w-0.5 h-6 transition-colors",
                          isCompleted ? "bg-green-500" : "bg-border"
                        )} />
                      )}
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Alternatives */}
      {plan.alternatives && plan.alternatives.length > 0 && (
        <Card className="border-blue-500/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-blue-500" />
              <span>Alternative Approaches</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {plan.alternatives.map((alt, index) => (
                <Card
                  key={index}
                  className={cn(
                    "cursor-pointer transition-all hover:shadow-lg border-2",
                    selectedAlternative === index
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  )}
                  onClick={() => handleSelectAlternative(index)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <p className="font-semibold capitalize">{alt.approach}</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {alt.description}
                        </p>
                      </div>
                      <Badge variant="outline">
                        {Math.round(alt.confidence * 100)}%
                      </Badge>
                    </div>
                    <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="w-3 h-3" />
                      <span>{alt.steps.length} steps</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Edit Plan Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Execution Plan</DialogTitle>
            <DialogDescription>
              Modify the plan steps before execution
            </DialogDescription>
          </DialogHeader>
          {editedPlan && (
            <Tabs defaultValue="steps" className="w-full">
              <TabsList>
                <TabsTrigger value="steps">Steps</TabsTrigger>
                <TabsTrigger value="reasoning">Reasoning</TabsTrigger>
              </TabsList>
              <TabsContent value="steps" className="space-y-4">
                <div className="space-y-3">
                  {editedPlan.steps.map((step, index) => (
                    <Card key={index} className="p-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-sm">{index + 1}.</span>
                          <Textarea
                            value={step.description || ''}
                            onChange={(e) => {
                              const newPlan = { ...editedPlan }
                              newPlan.steps[index].description = e.target.value
                              setEditedPlan(newPlan)
                            }}
                            className="flex-1 min-h-[60px]"
                            placeholder="Step description"
                          />
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Badge variant="secondary">{step.action}</Badge>
                          {step.target_id && <span>Target: {step.target_id}</span>}
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              </TabsContent>
              <TabsContent value="reasoning" className="space-y-4">
                <Textarea
                  value={editedPlan.reasoning || ''}
                  onChange={(e) => {
                    setEditedPlan({ ...editedPlan, reasoning: e.target.value })
                  }}
                  className="min-h-[200px]"
                  placeholder="AI reasoning for this plan"
                />
              </TabsContent>
            </Tabs>
          )}
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" onClick={() => setShowEditDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveEdit}>
              Save Changes
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default CloudPlannerView

