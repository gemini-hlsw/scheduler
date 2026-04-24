import { Link } from "react-router-dom";
import { useLocation } from "react-router-dom";

import type { JSX } from "react";
import { cn } from "../../../lib/utils";

interface CustomLink {
  to: string;
  name: string;
}

export default function Navbar() {
  const location = useLocation();

  const links: Array<CustomLink>[] = [
    [{ to: "/operation", name: "Operation" }],
    [
      { to: "/validation", name: "Validation" },
      { to: "/simulation", name: "Simulation" },
    ],
  ];

  const groups: JSX.Element[] = [];
  links?.map((group, i) => {
    const buttons: JSX.Element[] = [];
    group?.map((button, j) => {
      buttons.push(
        <Link
          className={cn(
            location.pathname === button.to
              ? "bg-white dark:bg-black"
              : "dark:bg-black/30 bg-white/30",
            "md:[writing-mode:vertical-lr] [writing-mode:initial] md:-scale-100 px-2.5",
            "dark:hover:text-fuchsia-400 dark:hover:bg-black/60",
            "hover:text-fuchsia-800 hover:bg-white/60",
            "transition-colors duration-300"
          )}
          to={button.to}
          key={`link_${j}`}
        >
          {button.name}
        </Link>
      );
    });
    groups.push(
      <div
        className={cn("md:mt-5 flex md:flex-col flex-row gap-0")}
        key={`group_${i}`}
      >
        {buttons}
      </div>
    );
  });

  return (
    <nav
      className={cn(
        "order-2 md:order-1",
        "dark:bg-white/20 bg-black/20",
        "flex md:flex-col gap-5"
      )}
    >
      {groups}
    </nav>
  );
}
