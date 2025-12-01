import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

export const MetricCard = ({ title, value, icon, trend, className }) => {
  const isPositive = trend?.startsWith('+') || trend === 'stable'
  const isNegative = trend?.startsWith('-')
  
  return (
    <Card 
      className={cn(
        "group relative overflow-hidden",
        "bg-gradient-to-br from-card to-card/80",
        "border-primary/20 hover:border-primary/40",
        "shadow-md hover:shadow-xl hover:shadow-primary/10",
        "transition-all duration-300 ease-out",
        "hover:-translate-y-1",
        "before:absolute before:inset-0 before:bg-gradient-to-br before:from-primary/5 before:to-transparent before:opacity-0 hover:before:opacity-100 before:transition-opacity before:duration-300",
        className
      )}
    >
      <CardContent className="p-6 relative z-10">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-muted-foreground mb-2 uppercase tracking-wide">
              {title}
            </p>
            <p className="text-3xl font-bold text-foreground mb-2 tabular-nums">
              {value}
            </p>
            <div className="flex items-center gap-1.5">
              <p className={cn(
                "text-xs font-semibold",
                isPositive && "text-green-500 dark:text-green-400",
                isNegative && "text-red-500 dark:text-red-400",
                !isPositive && !isNegative && "text-muted-foreground"
              )}>
                {trend}
              </p>
              {isPositive && !trend?.includes('stable') && (
                <span className="text-green-500">↗</span>
              )}
              {isNegative && (
                <span className="text-red-500">↘</span>
              )}
            </div>
          </div>
          <div className={cn(
            "flex-shrink-0 p-3 rounded-xl",
            "bg-primary/10 group-hover:bg-primary/20",
            "text-primary group-hover:text-primary",
            "transition-all duration-300",
            "group-hover:scale-110 group-hover:rotate-3"
          )}>
            {icon}
          </div>
        </div>
      </CardContent>
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/0 via-primary/0 to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
    </Card>
  )
}
