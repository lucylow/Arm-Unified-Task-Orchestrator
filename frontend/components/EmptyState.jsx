import { AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'

export const EmptyState = ({ onStartTask }) => (
  <div className="text-center py-12">
    <AlertCircle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
    <h3 className="font-semibold text-foreground mb-2">No Active Tasks</h3>
    <p className="text-muted-foreground mb-6">Start a new automation task to see live execution</p>
    {onStartTask && (
      <Button 
        onClick={onStartTask}
        className="bg-primary text-primary-foreground hover:bg-primary/90"
      >
        Start New Task
      </Button>
    )}
  </div>
)
