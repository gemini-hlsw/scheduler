import React, { createContext, useState, ReactNode } from "react";

// ------------------------------------------------------------
// Populate initial data, just for testing, should be removed
import { NightPlanType, RunSummary } from "../../types";
import { NightPlansWithEvent } from "@/gql/graphql";
// ------------------------------------------------------------

interface GlobalStateContextType {
  connectionState: { name: string; isConnected: boolean | null };
  setConnectionState: React.Dispatch<
    React.SetStateAction<{ name: string; isConnected: boolean | null }>
  >;
  nightPlans: NightPlanType[];
  setNightPlans: React.Dispatch<React.SetStateAction<NightPlanType[]>>;
  rtPlan: NightPlansWithEvent;
  setRtPlan: React.Dispatch<React.SetStateAction<NightPlansWithEvent>>;
  plansSummary: RunSummary;
  setPlansSummary: React.Dispatch<React.SetStateAction<RunSummary>>;
  thesis: number;
  setThesis: React.Dispatch<React.SetStateAction<number>>;
  power: number;
  setPower: React.Dispatch<React.SetStateAction<number>>;
  metPower: number;
  setMetPower: React.Dispatch<React.SetStateAction<number>>;
  visPower: number;
  setVisPower: React.Dispatch<React.SetStateAction<number>>;
  whaPower: number;
  setWhaPower: React.Dispatch<React.SetStateAction<number>>;
  airPower: number;
  setAirPower: React.Dispatch<React.SetStateAction<number>>;
  semesterVisibility: boolean;
  setSemesterVisibility: React.Dispatch<React.SetStateAction<boolean>>;
  loadingPlan: boolean;
  setLoadingPlan: React.Dispatch<React.SetStateAction<boolean>>;
  imageQuality: number;
  setImageQuality: React.Dispatch<React.SetStateAction<number>>;
  cloudCover: number;
  setCloudCover: React.Dispatch<React.SetStateAction<number>>;
  windDirection: number;
  setWindDirection: React.Dispatch<React.SetStateAction<number>>;
  windSpeed: number;
  setWindSpeed: React.Dispatch<React.SetStateAction<number>>;
  uuid: string;
}

export const GlobalStateContext = createContext<GlobalStateContextType>(null!);

export default function GlobalStateProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [connectionState, setConnectionState] = useState({
    name: "",
    isConnected: null,
  });
  const [nightPlans, setNightPlans] = useState<NightPlanType[]>([]);
  const [rtPlan, setRtPlan] = useState<NightPlansWithEvent>(
    {} as NightPlansWithEvent,
  );
  const [plansSummary, setPlansSummary] = useState<RunSummary>({
    summary: {},
    metricsPerBand: {},
  });
  const [thesis, setThesis] = useState(1.1);
  const [power, setPower] = useState(2);
  const [metPower, setMetPower] = useState(1.0);
  const [visPower, setVisPower] = useState(1.0);
  const [whaPower, setWhaPower] = useState(1.0);
  const [airPower, setAirPower] = useState(0.0);
  const [imageQuality, setImageQuality] = useState(0.7);
  const [cloudCover, setCloudCover] = useState(0.7);
  const [windDirection, setWindDirection] = useState(20);
  const [windSpeed, setWindSpeed] = useState(10);
  const [semesterVisibility, setSemesterVisibility] = useState(false);
  const [loadingPlan, setLoadingPlan] = useState(false);
  const [uuid] = useState(
    new Date()
      .toISOString()
      .substring(0, 16)
      .replaceAll("-", "")
      .replace(":", ""),
  );

  return (
    <GlobalStateContext.Provider
      value={{
        connectionState,
        setConnectionState,
        nightPlans,
        setNightPlans,
        rtPlan,
        setRtPlan,
        plansSummary,
        setPlansSummary,
        thesis,
        setThesis,
        power,
        setPower,
        metPower,
        setMetPower,
        visPower,
        setVisPower,
        whaPower,
        setWhaPower,
        airPower,
        setAirPower,
        semesterVisibility,
        setSemesterVisibility,
        loadingPlan,
        setLoadingPlan,
        uuid,
        imageQuality,
        setImageQuality,
        cloudCover,
        setCloudCover,
        windDirection,
        setWindDirection,
        windSpeed,
        setWindSpeed,
      }}
    >
      {children}
    </GlobalStateContext.Provider>
  );
}
