import * as React from 'react';
import { LineChart, lineElementClasses } from '@mui/x-charts/LineChart';
import { BarChart } from '@mui/x-charts/BarChart';
import { PieChart } from '@mui/x-charts/PieChart';
import { ScatterChart } from '@mui/x-charts/ScatterChart';

const width = 600;
const height = 400;
 
export function ChartComponent({ data }) {
  const commonProps = {
    width,
    height,
    slotProps: {
      legend: {
        labelStyle: {
          fontSize: 18,
          fill: 'white',
        },
      },
    },
    sx: {
      backgroundColor: 'black',
      '& .MuiChartsAxis-label': { fill: 'white' },
      '& .MuiChartsLegend-label': { fill: 'white' },
      '& .MuiChartsAxis-line': { stroke: 'white' },
      '& .MuiChartsAxis-tick': { stroke: 'white' },
    },
  };

  if (data.chartType === 'bar') {
    return <BarChart {...commonProps} xAxis={[{ scaleType: data.xAxisType || 'point', data: data.labels }]} series={data.datasets} />;
  } else if (data.chartType === 'line') {
    return <LineChart {...commonProps} xAxis={[{ scaleType: data.xAxisType || 'point', data: data.labels }]} series={data.datasets} />;
  } else if (data.chartType === 'pie') {
    return <PieChart {...commonProps} series={[{ data: data.datasets }]} />;
  } else if (data.chartType === 'scatter') {
    return <ScatterChart {...commonProps} series={data.datasets} />;
  } else {
    return <div>Unsupported chart type</div>;
  }
}

export function CompetitorRevenueChart() {
  return <ChartComponent data={{
    ...competitorData,
    chartType: 'bar',
    xAxisType: 'band',
  }} />;
}
