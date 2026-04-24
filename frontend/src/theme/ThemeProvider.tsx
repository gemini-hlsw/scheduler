import { createContext, useState, ReactNode, useEffect } from "react";
import { Theme } from "../types";

interface ThemeContextType {
  theme: Theme;
  toggleTheme(): void;
}

export const ThemeContext = createContext<ThemeContextType>(null!);

export default function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>("dark");

  function toggleTheme() {
    setTheme(theme === "dark" ? "light" : "dark");
  }

  const value = { theme, toggleTheme };

  useEffect(() => {
    document.body.classList.value = theme;
  }, [theme]);

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}
