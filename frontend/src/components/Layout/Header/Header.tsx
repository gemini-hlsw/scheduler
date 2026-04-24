import MainTitle from "./MainTitle";
import { useContext } from "react";
import { ThemeContext } from "../../../theme/ThemeProvider";
import { About } from "./About";
// import "./Header.scss";
import { GlobalStateContext } from "../../GlobalState/GlobalState";
import { cn } from "../../../lib/utils";
import { Button } from "@/components/ui/button";
import { FaMoon, FaSun } from "react-icons/fa";
import { ConnectionBadge } from "@/components/ui/connectionBadge";

export default function Header({ title }: { title: string }) {
  const { theme, toggleTheme } = useContext(ThemeContext);
  const { connectionState } = useContext(GlobalStateContext);

  return (
    <div
      className={cn(
        "flex flex-row items-center justify-between",
        "px-4 h-8 shrink-0 dark:bg-white/10 dark:text-white",
        "light:bg-black/10 light:text-black",
        "border-b border-gray-200 dark:border-gray-700"
      )}
    >
      <div className={cn("flex flex-row items-center shrink")}>
        <MainTitle title={title} />
      </div>
      <div className={cn("mx-auto")}>
        <div className="flex flex-row gap-2 grow items-center">
          {connectionState.isConnected !== null && (
            <ConnectionBadge
              isOnline={connectionState.isConnected}
              text={`ID: ${connectionState.name} | `}
              className="font-bold"
            />
          )}
        </div>
      </div>
      <div className={cn("flex flex-row items-center gap-2 shrink")}>
        <About />
        <Button variant="outline" size="xs" onClick={toggleTheme}>
          {theme === "dark" ? <FaMoon /> : <FaSun />}
          <span className="label">{theme}</span>
        </Button>
      </div>
    </div>
  );
}
