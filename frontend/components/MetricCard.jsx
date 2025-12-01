import { Card, CardContent } from '@/components/ui/card'

export const MetricCard = ({ title, value, icon, trend }) => (
  <Card className="shadow-md hover:shadow-lg transition-shadow border-primary/20">
    <CardContent className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground mb-1">{title}</p>
          <p className="text-2xl font-bold text-foreground">{value}</p>
          <p className="text-xs text-muted-foreground mt-1">{trend}</p>
        </div>
        {icon}
      </div>
    </CardContent>
  </Card>
)
