import { useMemo } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { CheckCircle2, Circle, ArrowDown, Play, Pause } from 'lucide-react'
import { cn } from '@/lib/utils'

export function PlanFlowDiagram({ plan, currentStep = 0, isExecuting = false, isPaused = false }) {
  if (!plan || !plan.steps) return null

  const steps = plan.steps

  return (
    <Card className="border-primary/20">
      <CardContent className="p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-lg">Execution Flow</h3>
            <Badge variant="outline">
              {currentStep} / {steps.length}
            </Badge>
          </div>
          
          <div className="relative">
            {/* Flow Diagram */}
            <div className="space-y-3">
              {steps.map((step, index) => {
                const isActive = currentStep === index + 1
                const isCompleted = currentStep > index + 1
                const isPending = currentStep < index + 1

                return (
                  <div key={step.id || `flow_${index}`} className="relative">
                    {/* Step Node */}
                    <div
                      className={cn(
                        "flex items-center gap-4 p-4 rounded-lg border-2 transition-all duration-300",
                        isActive && "bg-primary/10 border-primary shadow-lg shadow-primary/20 scale-105",
                        isCompleted && "bg-green-500/10 border-green-500/50",
                        isPending && "bg-muted/30 border-border/50 opacity-60"
                      )}
                    >
                      {/* Step Icon */}
                      <div className={cn(
                        "flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center transition-all",
                        isActive && "bg-primary text-primary-foreground ring-4 ring-primary/30",
                        isCompleted && "bg-green-500 text-white",
                        isPending && "bg-muted text-muted-foreground"
                      )}>
                        {isCompleted ? (
                          <CheckCircle2 className="w-6 h-6" />
                        ) : isActive ? (
                          isPaused ? (
                            <Pause className="w-6 h-6" />
                          ) : (
                            <Play className="w-6 h-6" />
                          )
                        ) : (
                          <Circle className="w-6 h-6" />
                        )}
                      </div>

                      {/* Step Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <p className={cn(
                            "font-semibold text-sm",
                            isActive && "text-primary",
                            isCompleted && "text-green-500",
                            isPending && "text-muted-foreground"
                          )}>
                            Step {index + 1}: {step.description || step.action}
                          </p>
                          {step.confidence && (
                            <Badge 
                              variant="outline" 
                              className={cn(
                                "text-xs",
                                step.confidence >= 0.9 && "border-green-500/30",
                                step.confidence >= 0.8 && step.confidence < 0.9 && "border-yellow-500/30",
                                step.confidence < 0.8 && "border-orange-500/30"
                              )}
                            >
                              {Math.round(step.confidence * 100)}%
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Badge variant="secondary" className="text-xs font-mono">
                            {step.action}
                          </Badge>
                          {step.target_id && (
                            <span>→ {step.target_id}</span>
                          )}
                        </div>
                      </div>

                      {/* Active Pulse Indicator */}
                      {isActive && !isPaused && (
                        <div className="absolute inset-0 rounded-lg bg-primary/5 animate-pulse" />
                      )}
                    </div>

                    {/* Connector Arrow */}
                    {index < steps.length - 1 && (
                      <div className="flex justify-center my-2">
                        <div className={cn(
                          "w-0.5 h-6 transition-colors",
                          isCompleted ? "bg-green-500" : "bg-border"
                        )} />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>

            {/* Progress Indicator */}
            {isExecuting && (
              <div className="mt-6 pt-4 border-t border-border">
                <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                  <span>Overall Progress</span>
                  <span className="font-semibold text-foreground">
                    {Math.round((currentStep / steps.length) * 100)}%
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                  <div
                    className={cn(
                      "h-full bg-gradient-to-r from-primary to-primary/80 transition-all duration-500 rounded-full",
                      currentStep === steps.length && "from-green-500 to-green-400"
                    )}
                    style={{ width: `${(currentStep / steps.length) * 100}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default PlanFlowDiagram

