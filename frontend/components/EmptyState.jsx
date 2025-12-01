import { AlertCircle, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export const EmptyState = ({ onStartTask, className }) => (
  <div className={cn("text-center py-16 px-4 animate-fade-in", className)}>
    <div className="relative inline-block mb-6">
      <div className="absolute inset-0 bg-primary/20 rounded-full blur-2xl animate-pulse" />
      <div className="relative p-4 rounded-full bg-muted/50 border border-primary/20">
        <AlertCircle className="w-12 h-12 text-muted-foreground" />
      </div>
    </div>
    <h3 className="text-xl font-semibold text-foreground mb-2">
      No Active Tasks
    </h3>
    <p className="text-muted-foreground mb-8 max-w-md mx-auto">
      Start a new automation task to see live execution and AI agent activity
    </p>
    {onStartTask && (
      <Button 
        onClick={onStartTask}
        className="bg-primary text-primary-foreground hover:bg-primary/90 hover:shadow-lg hover:shadow-primary/30 transition-all duration-300 hover:-translate-y-0.5"
        size="lg"
      >
        <Sparkles className="w-4 h-4 mr-2" />
        Start New Task
      </Button>
    )}
  </div>
)
