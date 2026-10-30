import React, { useContext, useEffect, useRef } from "react";
import { FaCloudMoonRain, FaCog } from "react-icons/fa";
import { renderToString } from "react-dom/server";
import Highcharts, { SVGRenderer, SeriesArearangeOptions } from "highcharts";
import HighchartsReact, {
  HighchartsReactRefObject,
} from "highcharts-react-official";
import HighchartMore from "highcharts/highcharts-more";
HighchartMore(Highcharts);

import { ThemeContext } from "../../theme/ThemeProvider";
import { getSiteOffset, utcToLocal } from "@/helpers/utcTime";
import { AltitudePoint, Event, TimeLossWindow } from "@/types";
import MoonPhaseIcon from "./MoonPhaseIcon";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const debounce = <F extends (...args: any[]) => void>(
  func: F,
  delay: number,
) => {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  return function (...args: Parameters<F>) {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      func(...args);
    }, delay);
  } as F;
};

// The x axis spans the whole night plus this margin on each side, so every
// plan of the night is drawn on the same axis whatever time it starts at.
const TWILIGHT_MARGIN = 30 * 60 * 1000;
const HOUR = 60 * 60 * 1000;
// Shades the margins, where the sun is above the 12° twilight.
const TWILIGHT_BAND_COLOR = "rgba(128, 128, 128, 0.25)";
const TWILIGHT_LINE_COLOR = "rgba(128, 128, 128, 0.9)";
const NIGHT_LIMIT_LINE_COLOR = "var(--color-amber-500)";
const MOON_LINE_COLOR = "#a3a3a3";

interface Visit {
  startDate: Date;
  endDate: Date;
  yPoints: number[];
  label: string;
  instrument: string;
  atomTimes: number[];
}

interface AltAzPlotProps {
  data: Visit[];
  event: Event;
  eveTwilight: string;
  mornTwilight: string;
  // The custom start and end of night, if any.
  nightStart?: string | null;
  nightEnd?: string | null;
  moonAltitude?: AltitudePoint[];
  // At the middle of the night: lit fraction, 0 (new) to 1 (full).
  moonIllumination?: number | null;
  moonWaxing?: boolean | null;
  site: string;
  closureWindows: TimeLossWindow[];
}

const AltAzPlot: React.FC<AltAzPlotProps> = ({
  data,
  event,
  eveTwilight,
  mornTwilight,
  nightStart,
  nightEnd,
  moonAltitude = [],
  moonIllumination,
  moonWaxing,
  site,
  closureWindows,
}) => {
  const INITIAL_TIMEZONE = getSiteOffset(site);

  const weatherLossIconHtml = renderToString(
    <FaCloudMoonRain
      style={{ width: "577px", height: "512px" }}
      width={10}
      className="tex-cyan-500 scale-5 absolute -top-60 -left-68.5 origin-center"
    />,
  );

  const faultLossIconHtml = renderToString(
    <FaCog
      style={{ width: "577px", height: "512px" }}
      width={10}
      className="tex-cyan-500 scale-5 absolute -top-60 -left-68.5 origin-center"
    />,
  );

  // Get theme context to modify chart values
  const { theme } = useContext(ThemeContext);
  const textColor = theme === "light" ? "#000" : "#ECEAEA";
  const gridLineColor = theme === "light" ? "#e6e6e6" : "#444";

  // ref for post-render use
  const chartRef = useRef<HighchartsReactRefObject>(null);
  const labelRef = useRef(null);

  // Array of colors from Highcharts
  const colorsOption = Highcharts.getOptions().colors;
  // Highcharts default palette only has 10 colors, extend it so every
  // instrument gets its own color
  const extraColors: Highcharts.ColorString[] = [
    "#f2c80f",
    "#a0522d",
    "#ff6fb5",
    "#7f8c8d",
    "#808000",
    "#8b0a50",
    "#c6e377",
  ];
  const colors = [
    ...(colorsOption
      ? colorsOption.filter(
          (color): color is Highcharts.ColorString => typeof color === "string",
        )
      : []),
    ...extraColors,
  ];
  // Used for instruments not listed below
  const fallbackColor: Highcharts.ColorString = "#999999";

  const instruments = [
    "GMOS-N",
    "GMOS-S",
    "GNIRS",
    "NIRI",
    "Flamingos2",
    "GSAOI",
    "GPI",
    "IGRINS",
    "IGRINS-2",
    "NIFS",
    "Alopeke",
    "MAROON-X",
    "GHOST",
  ];

  type ColorMap = {
    [key: string]: Highcharts.ColorString;
  };
  const createMap = (
    keys: string[],
    colors: Highcharts.ColorString[],
  ): ColorMap => {
    const map: ColorMap = {};

    for (let i = 0; i < keys.length; i++) {
      map[keys[i]] = colors[i];
    }

    return map;
  };
  const colorMap = createMap(instruments, colors);
  const eveTwiDate = new Date(eveTwilight);
  const mornTwiDate = new Date(mornTwilight);
  const eveTwiX = utcToLocal(eveTwiDate, INITIAL_TIMEZONE);
  const mornTwiX = utcToLocal(mornTwiDate, INITIAL_TIMEZONE);
  const axisMin = eveTwiX - TWILIGHT_MARGIN;
  const axisMax = mornTwiX + TWILIGHT_MARGIN;

  const seriesData: Array<SeriesArearangeOptions> = data.map((d: Visit) => {
    const instrumentColor = colorMap[d.instrument] ?? fallbackColor;
    const zones = [];
    for (let i = 0; i < d.atomTimes.length; i++) {
      const atomTime = d.atomTimes[i];
      const atomX = utcToLocal(
        new Date(d.startDate.getTime() + atomTime * 60 * 1000),
        INITIAL_TIMEZONE,
      );
      zones.push({
        value: atomX,
        color: instrumentColor,
        fillColor:
          i % 2 === 0 ? `${instrumentColor}66` : `${instrumentColor}33`, // 20% opacity for even atoms, 0% for odd
      });
    }

    const yMinArray = d.yPoints.map(() => 0);
    return {
      name: d.instrument,
      type: "arearange",
      zoneAxis: "x",
      zones: zones,
      data: d.yPoints.map((y: number, i: number) => {
        return {
          x: utcToLocal(d.startDate, INITIAL_TIMEZONE) + i * 60 * 1000,
          low: yMinArray[i],
          high: y,
        };
      }),
      lineWidth: 1,
      color: instrumentColor,
      fillOpacity: 0.3,
      zIndex: 0,
      marker: {
        enabled: false,
      },
      events: {
        legendItemClick: function () {
          return false; // Prevents the default action, which is toggling visibility
        },
      },
      showInLegend: true, // Hide this series in the legend
    };
  });

  // Create a set to keep track of names we've seen
  const seenNames = new Set();

  // Modify the series to set showInLegend to false for duplicates
  for (const serie of seriesData) {
    if (seenNames.has(serie.name)) {
      serie.showInLegend = false; // Don't show in legend if it's a duplicate
    } else {
      seenNames.add(serie.name); // Add the name to the set if we haven't seen it
    }
  }

  // Below the horizon is outside the y axis, so only the part above it shows.
  const moonSeries: Highcharts.SeriesLineOptions = {
    id: "moon",
    name: "Moon",
    type: "line",
    data: moonAltitude.map(({ time, altitude }) => [
      utcToLocal(new Date(time), INITIAL_TIMEZONE),
      altitude,
    ]),
    color: MOON_LINE_COLOR,
    dashStyle: "ShortDash",
    lineWidth: 1.5,
    zIndex: 1,
    marker: { enabled: false },
    enableMouseTracking: false,
    events: {
      legendItemClick: function () {
        return false;
      },
    },
  };
  // Legend entry for the moon: its phase and how much of it is lit.
  const moonLegendHtml =
    moonIllumination == null
      ? "Moon"
      : `<span style="display: inline-flex; align-items: center; gap: 4px;">${renderToString(
          <MoonPhaseIcon
            illumination={moonIllumination}
            waxing={moonWaxing ?? true}
            southern={site === "GS"}
          />,
        )}Moon (${Math.round(moonIllumination * 100)}%)</span>`;

  const allSeries: Highcharts.SeriesOptionsType[] = [
    ...seriesData,
    moonSeries,
  ];

  function updateChart() {
    const chart = chartRef.current.chart;

    chart.update({
      series: allSeries,
    });

    // Draw event line to recognize the time when it occurs
    chart.xAxis[0].update({
      plotBands: [],
      plotLines: [],
    });

    // Remove old labels
    labelRef.current.forEach((lbl: SVGRenderer) => lbl.destroy());

    // Render custom labels for each section
    data.forEach((d) => {
      const x =
        (utcToLocal(d.startDate, INITIAL_TIMEZONE) +
          utcToLocal(d.endDate, INITIAL_TIMEZONE)) /
        2;

      const xPos = chart.xAxis[0].toPixels(x, false);

      const lbl = chart.renderer
        .text(d.label, xPos + 4.5, 350)
        .attr({
          rotation: -90,
          textAlign: "center",
        })
        .css({
          fontWeight: "bold",
          color: textColor, // Change the color of the custom label
        })
        .add();

      labelRef.current.push(lbl);
    });

    chart.xAxis[0].addPlotLine({
      id: "event",
      color: "var(--color-blue-500)",
      width: 2,
      value: utcToLocal(new Date(event.time), INITIAL_TIMEZONE),
      zIndex: 100,
    });

    // Mark the twilights and, when given, the custom start and end of night
    const limitLines = [
      {
        id: "evening-twilight",
        label: "Eve. twilight",
        time: eveTwilight,
        color: TWILIGHT_LINE_COLOR,
      },
      {
        id: "morning-twilight",
        label: "Morn. twilight",
        time: mornTwilight,
        color: TWILIGHT_LINE_COLOR,
      },
      {
        id: "night-start",
        label: "Night start",
        time: nightStart,
        color: NIGHT_LIMIT_LINE_COLOR,
      },
      {
        id: "night-end",
        label: "Night end",
        time: nightEnd,
        color: NIGHT_LIMIT_LINE_COLOR,
      },
    ];
    limitLines.forEach(({ id, label, time, color }) => {
      if (!time) return;
      chart.xAxis[0].addPlotLine({
        id,
        color,
        width: 1.5,
        dashStyle: "Dash",
        value: utcToLocal(new Date(time), INITIAL_TIMEZONE),
        zIndex: 5,
        label: {
          text: label,
          rotation: 90,
          align: "left",
          x: 3,
          y: 4,
          style: { color: textColor, fontSize: "10px" },
        },
      });
    });

    // Shade outside the twilights
    chart.xAxis[0].addPlotBand({
      id: "before-evening-twilight",
      from: axisMin,
      to: eveTwiX,
      color: TWILIGHT_BAND_COLOR,
    });
    chart.xAxis[0].addPlotBand({
      id: "after-morning-twilight",
      from: mornTwiX,
      to: axisMax,
      color: TWILIGHT_BAND_COLOR,
    });

    // Add possible closure zones
    closureWindows.forEach((window) => {
      chart.xAxis[0].addPlotBand({
        id: `${window.lossType}-${window.start}`,
        from: utcToLocal(new Date(window.start), INITIAL_TIMEZONE),
        to: window.end
          ? utcToLocal(new Date(window.end), INITIAL_TIMEZONE)
          : mornTwiX,
        color:
          window.lossType === "weather"
            ? "rgba(50, 200, 230, 0.3)"
            : "rgba(200, 0, 0, 0.3)",
        zIndex: 10,
        label: {
          useHTML: true,
          text: `<div style="width: 30px; height: 30px;">${window.lossType === "weather" ? weatherLossIconHtml : faultLossIconHtml}</div>`,
        },
      });
    });
    chart.xAxis[0].addPlotBand({});

    chart.redraw();
  }

  useEffect(() => {
    if (!labelRef.current) {
      labelRef.current = [];
    }

    if (chartRef.current) {
      updateChart();
    }
  }, [seriesData]);

  const debouncedResizeHandler = debounce(updateChart, 300);

  useEffect(() => {
    // Set initial size if not already set (for SSR compatibility)
    window.addEventListener("resize", debouncedResizeHandler);

    return () => {
      window.removeEventListener("resize", debouncedResizeHandler);
    };
  }, [debouncedResizeHandler, updateChart]);

  const options: Highcharts.Options = {
    time: {
      timezone: "Pacific/Honolulu",
    },
    chart: {
      type: "arearange",
    },

    tooltip: {
      // Use the shared tooltip to show information for the entire area
      shared: true,
      formatter: function () {
        if (this.points) {
          // Assuming the first point is representative for the area
          const point = this.series.name;
          return point;
        }
        return false; // No tooltip for individual points
      },
    },
    title: {
      text: undefined,
    },
    xAxis: {
      type: "datetime",
      labels: {
        formatter: function () {
          return Highcharts.dateFormat("%H:%M", this.value as number);
        },
        style: {
          color: textColor, // Change the color of y-axis tick labels
        },
      },
      min: axisMin,
      max: axisMax,
      startOnTick: false,
      endOnTick: false,
      // Every hour on the hour.
      tickPositioner: function () {
        const positions = [];
        for (
          let i = Math.ceil(axisMin / HOUR) * HOUR;
          i <= axisMax;
          i += HOUR
        ) {
          positions.push(i);
        }
        return positions;
      },
    },
    yAxis: {
      title: {
        text: null,
      },
      labels: {
        style: {
          color: textColor, // Change the color of y-axis tick labels
        },
      },
      gridLineColor: gridLineColor, // Change the color of horizontal grid lines
      min: 0,
      max: 90,
    },
    legend: {
      enabled: true,
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      // For the moon phase icon.
      useHTML: true,
      labelFormatter: function () {
        const series = this as Highcharts.Series;
        return series.options.id === "moon" ? moonLegendHtml : series.name;
      },
      itemStyle: {
        color: textColor,
      },
    },
    series: allSeries,
  };

  return (
    <div className="scheduler-plot">
      <HighchartsReact
        highcharts={Highcharts}
        options={options}
        ref={chartRef}
      />
    </div>
  );
};

export default AltAzPlot;
